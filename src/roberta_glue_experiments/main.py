import argparse
import os
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import itertools
from tqdm import tqdm
import cma
import numpy as np
import pandas as pd
import logging

from transformers import (
    RobertaTokenizerFast,
    RobertaModel,
    default_data_collator,
)
from datasets import load_dataset
from evaluationpp import TaskEvaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastGPUTaskVectorPool:
    def __init__(self, base_model_path, task_names, task_model_paths, device="cuda"):
        self.device = device
        self.task_names = task_names
        print(f"Loading Base Model from {base_model_path} to {device}...")

        self.base_model = RobertaModel.from_pretrained(base_model_path).to(device)
        self.base_model.eval()
        self.base_model.requires_grad_(False)
        self.base_state_dict = self.base_model.state_dict()

        self.num_layers = 0
        for key in self.base_state_dict.keys():
            match = re.search(r'encoder\.layer\.(\d+)\.', key)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx > self.num_layers:
                    self.num_layers = layer_idx
        self.num_layers += 1
        print(f"Detected {self.num_layers} layers.")

        self.task_deltas = []
        print("Computing Deltas and keeping them on GPU...")

        base_sd_cpu = {k: v.cpu() for k, v in self.base_state_dict.items()}

        for t_name, t_path in zip(task_names, task_model_paths):
            print(f"  - Processing Delta for {t_name}...")
            ft_model = RobertaModel.from_pretrained(t_path)
            ft_sd = ft_model.state_dict()

            delta = {}
            for k, v in ft_sd.items():
                if k in base_sd_cpu:
                    if v.is_floating_point():
                        d = v - base_sd_cpu[k]
                        delta[k] = d.to(device)

            self.task_deltas.append(delta)
            del ft_model, ft_sd

        del base_sd_cpu
        torch.cuda.empty_cache()
        print(f"All Deltas loaded on GPU.")

    def get_layer_index(self, key):
        match = re.search(r'encoder\.layer\.(\d+)\.', key)
        return int(match.group(1)) if match else self.num_layers

    def apply_coeffs(self, coeffs, mode="task_wise"):
        num_tasks = len(self.task_deltas)
        num_groups = self.num_layers + 1

        modified_keys = set()
        for d in self.task_deltas:
            modified_keys.update(d.keys())

        with torch.no_grad():
            for k in modified_keys:
                base_param = self.base_state_dict[k]
                total_delta = torch.zeros_like(base_param)

                for t_idx in range(num_tasks):
                    if k not in self.task_deltas[t_idx]:
                        continue

                    if mode == "task_wise":
                        c = coeffs[t_idx]
                    else:
                        l_idx = self.get_layer_index(k)
                        c = coeffs[t_idx * num_groups + l_idx]

                    if abs(c) > 1e-6:
                        total_delta.add_(self.task_deltas[t_idx][k], alpha=c)

                base_param.add_(total_delta)

    def restore_base(self, coeffs, mode="task_wise"):
        num_tasks = len(self.task_deltas)
        num_groups = self.num_layers + 1

        modified_keys = set()
        for d in self.task_deltas:
            modified_keys.update(d.keys())

        with torch.no_grad():
            for k in modified_keys:
                base_param = self.base_state_dict[k]
                total_delta = torch.zeros_like(base_param)

                for t_idx in range(num_tasks):
                    if k not in self.task_deltas[t_idx]:
                        continue
                    if mode == "task_wise":
                        c = coeffs[t_idx]
                    else:
                        l_idx = self.get_layer_index(k)
                        c = coeffs[t_idx * num_groups + l_idx]

                    if abs(c) > 1e-6:
                        total_delta.add_(self.task_deltas[t_idx][k], alpha=c)

                base_param.sub_(total_delta)


class CalibrationDataManager:
    def __init__(self, tokenizer, task_names, batch_size=8, max_length=128):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.loaders = {}
        self.task_key_map = {
            "cola": ("sentence", None), "sst2": ("sentence", None),
            "mrpc": ("sentence1", "sentence2"), "mnli": ("premise", "hypothesis"),
            "qnli": ("question", "sentence"), "rte": ("sentence1", "sentence2"),
            "stsb": ("sentence1", "sentence2")
        }

        print("Initializing Calibration Data Loaders for Search...")
        for t in task_names:
            self._init_loader(t)

    def _init_loader(self, t):
        t_lower = t.lower()
        if t_lower in self.task_key_map:
            dataset = load_dataset("glue", t_lower, split="validation")
        elif t_lower == "race":
            try:
                dataset = load_dataset("race", "all", split="validation")
            except:
                dataset = load_dataset("race", "high", split="validation")
        elif t_lower == "squad":
            dataset = load_dataset("squad", split="validation")
        else:
            print(f"Task {t} not supported for calibration.")
            return

        if len(dataset) > 2048:
            dataset = dataset.select(range(2048))

        def tokenize_fn(examples):
            if t_lower == "squad":
                inputs = self.tokenizer(
                    examples["question"],
                    examples["context"],
                    truncation="only_second",
                    max_length=384,
                    stride=128,
                    padding="max_length",
                    return_offsets_mapping=True
                )
                offset_mapping = inputs.pop("offset_mapping")
                answers = examples["answers"]
                start_positions = []
                end_positions = []

                for i, offset in enumerate(offset_mapping):
                    answer = answers[i]
                    if len(answer["answer_start"]) == 0:
                        start_positions.append(0)
                        end_positions.append(0)
                        continue

                    start_char = answer["answer_start"][0]
                    end_char = answer["answer_start"][0] + len(answer["text"][0])
                    sequence_ids = inputs.sequence_ids(i)

                    idx = 0
                    while sequence_ids[idx] != 1:
                        idx += 1
                    context_start = idx
                    while sequence_ids[idx] == 1:
                        idx += 1
                    context_end = idx - 1

                    if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                        start_positions.append(0)
                        end_positions.append(0)
                    else:
                        idx = context_start
                        while idx <= context_end and offset[idx][0] <= start_char:
                            idx += 1
                        start_positions.append(idx - 1)
                        idx = context_end
                        while idx >= context_start and offset[idx][1] >= end_char:
                            idx -= 1
                        end_positions.append(idx + 1)

                inputs["start_positions"] = start_positions
                inputs["end_positions"] = end_positions
                return inputs

            elif t_lower == "race":
                first = [[c] * 4 for c in examples["article"]]
                second = [[f"{q} {o}" for o in opts] for q, opts in zip(examples["question"], examples["options"])]
                flat_first = sum(first, [])
                flat_second = sum(second, [])
                tokenized = self.tokenizer(flat_first, flat_second, truncation=True, max_length=self.max_length, padding="max_length")
                data = {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized.items()}
                label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                data["label"] = [label_map[a] for a in examples["answer"]]
                return data

            else:
                k1, k2 = self.task_key_map[t_lower]
                tokenized = self.tokenizer(examples[k1], examples[k2] if k2 else None, truncation=True, max_length=self.max_length, padding="max_length")
                if "label" in examples:
                    tokenized["labels"] = examples["label"]
                elif "labels" in examples:
                    tokenized["labels"] = examples["labels"]
                return tokenized

        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
        tokenized.set_format("torch")

        if t_lower == "race":
            def race_collate(features):
                label_name = "label" if "label" in features[0].keys() else "labels"
                labels = [feature.pop(label_name) for feature in features]
                batch_size = len(features)
                num_choices = len(features[0]["input_ids"])
                flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
                flattened_features = sum(flattened_features, [])
                batch = self.tokenizer.pad(flattened_features, padding=True, max_length=self.max_length, return_tensors="pt")
                batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
                batch["labels"] = torch.tensor(labels, dtype=torch.int64)
                return batch
            collate_fn = race_collate
        else:
            collate_fn = default_data_collator

        loader = DataLoader(tokenized, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        self.loaders[t_lower] = itertools.cycle(loader)

    def get_sample_batch(self, task_name):
        t_lower = task_name.lower()
        if t_lower not in self.loaders:
            return None
        return next(self.loaders[t_lower])

    def get_accumulated_batch(self, task_name, num_batches=4):
        """Concatenate multiple batches to reduce variance."""
        t_lower = task_name.lower()
        if t_lower not in self.loaders:
            return None

        batches = []
        for _ in range(num_batches):
            try:
                b = next(self.loaders[t_lower])
                batches.append(b)
            except StopIteration:
                self._init_loader(task_name)
                b = next(self.loaders[t_lower])
                batches.append(b)

        if not batches:
            return None

        keys = batches[0].keys()
        accumulated = {}
        for k in keys:
            if k in ['input_ids', 'attention_mask', 'labels', 'start_positions', 'end_positions']:
                tensors = [b[k] for b in batches]
                accumulated[k] = torch.cat(tensors, dim=0)
            else:
                accumulated[k] = batches[0][k]

        print(f"Task {task_name}: Accumulated {len(batches)} batches. Total Size: {accumulated['input_ids'].size(0)}")
        return accumulated


class AWA_Engine:
    def __init__(self, pool, tokenizer, task_names, heads_cache, batch_size):
        self.pool = pool
        self.data_manager = CalibrationDataManager(tokenizer, task_names, batch_size=batch_size)
        self.task_heads = heads_cache
        self.task_names = task_names
        self.num_labels_map = {"cola": 2, "sst2": 2, "mrpc": 2, "rte": 2, "stsb": 1, "squad": 0, "race": 4, "mnli": 3, "qnli": 2}

    def forward_loss(self, task_name, batch):
        task_lower = task_name.lower()
        head = self.task_heads.get(task_lower)
        if head is None:
            return 0.0

        input_ids = batch['input_ids'].to(self.pool.device)
        attention_mask = batch['attention_mask'].to(self.pool.device)

        if task_lower == "race":
            b_size, num_choices, seq_len = input_ids.shape
            flat_input_ids = input_ids.view(-1, seq_len)
            flat_mask = attention_mask.view(-1, seq_len)
            outputs = self.pool.base_model(input_ids=flat_input_ids, attention_mask=flat_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
        else:
            outputs = self.pool.base_model(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]

        if task_lower == "squad":
            logits = F.linear(outputs.last_hidden_state, head['qa_outputs.weight'], head['qa_outputs.bias'])
            start_logits, end_logits = logits.split(1, dim=-1)
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            start_loss = loss_fct(start_logits.squeeze(-1), batch['start_positions'].to(self.pool.device))
            end_loss = loss_fct(end_logits.squeeze(-1), batch['end_positions'].to(self.pool.device))
            return (start_loss + end_loss) / 2.0

        elif task_lower == "race":
            x = cls_output
            if 'classifier.dense.weight' in head:
                x = F.linear(x, head['classifier.dense.weight'], head['classifier.dense.bias'])
                x = torch.tanh(x)

            if 'classifier.out_proj.weight' in head:
                logits = F.linear(x, head['classifier.out_proj.weight'], head['classifier.out_proj.bias'])
            elif 'classifier.weight' in head:
                logits = F.linear(x, head['classifier.weight'], head['classifier.bias'])
            else:
                raise KeyError(f"Task {task_name}: Head weights not found. Available keys: {list(head.keys())}")

            return nn.CrossEntropyLoss()(logits.view(-1, 4), batch['labels'].to(self.pool.device))

        elif task_lower == "stsb":
            x = F.linear(cls_output, head['classifier.dense.weight'], head['classifier.dense.bias'])
            x = torch.tanh(x)
            logits = F.linear(x, head['classifier.out_proj.weight'], head['classifier.out_proj.bias'])
            return nn.MSELoss()(logits.view(-1), batch['labels'].to(self.pool.device).view(-1))

        else:
            x = F.linear(cls_output, head['classifier.dense.weight'], head['classifier.dense.bias'])
            x = torch.tanh(x)
            logits = F.linear(x, head['classifier.out_proj.weight'], head['classifier.out_proj.bias'])
            return nn.CrossEntropyLoss()(logits.view(-1, self.num_labels_map[task_lower]), batch['labels'].to(self.pool.device).view(-1))

    def run(self, mode="task_wise", max_steps=30):
        num_tasks = len(self.pool.task_deltas)
        if mode == "task_wise":
            num_params = num_tasks
        else:
            num_params = num_tasks * (self.pool.num_layers + 1)

        print(f"Optimizing {num_params} params. Mode: {mode}")

        initial_val = 1.00
        x0 = [initial_val] * num_params
        sigma0 = 0.05
        bounds = [0.1, 2.0]
        opts = {
            'popsize': 6,
            'maxiter': max_steps,
            'verbose': 1,
            'bounds': bounds,
        }
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        batches = {}

        for t in self.task_names:
            if t.lower() in ['cola', 'mrpc', 'rte']:
                b = self.data_manager.get_accumulated_batch(t, num_batches=8)
            else:
                b = self.data_manager.get_accumulated_batch(t, num_batches=4)
            if b:
                batches[t] = b

        print("Calculating Initial Baseline Loss for Normalization...")
        self.pool.apply_coeffs(x0, mode=mode)
        baseline_losses = {}
        for t, batch in batches.items():
            l = self.forward_loss(t, batch)
            val = l.item() if isinstance(l, torch.Tensor) else l
            baseline_losses[t] = val + 1e-8
            print(f"  Task {t}: Baseline Loss = {val:.4f}")
        self.pool.restore_base(x0, mode=mode)

        best_fitness = 999.0
        best_coeffs = x0

        patience = 6
        trigger_times = 0
        min_delta = 1e-4

        pbar = tqdm(range(max_steps), desc="Evolving")

        for step in pbar:
            if es.stop():
                break

            solutions = es.ask()
            fitness_list = []

            for sol in solutions:
                sol = np.clip(sol, bounds[0], bounds[1])
                self.pool.apply_coeffs(sol, mode=mode)

                loss_score = 0.0
                for t, batch in batches.items():
                    l = self.forward_loss(t, batch)
                    curr_loss = l.item() if isinstance(l, torch.Tensor) else l
                    base_loss = baseline_losses[t]
                    relative_diff = (curr_loss - base_loss) / base_loss
                    if relative_diff > 0:
                        loss_score += relative_diff * 100.0
                    else:
                        loss_score += relative_diff * 1.0

                fitness_list.append(loss_score)
                self.pool.restore_base(sol, mode=mode)

            es.tell(solutions, fitness_list)
            es.disp()

            min_idx = np.argmin(fitness_list)
            current_best_val = fitness_list[min_idx]

            if current_best_val < (best_fitness - min_delta):
                best_fitness = current_best_val
                best_coeffs = solutions[min_idx]
                trigger_times = 0
            else:
                trigger_times += 1

            pbar.set_postfix({
                "BestFit": f"{best_fitness:.4f}",
                "Patience": f"{trigger_times}/{patience}"
            })

            if trigger_times >= patience:
                print(f"\n[Early Stopping] Triggered at step {step+1}. No improvement for {patience} steps.")
                break

        return best_coeffs


def load_checkpoint_state_dict(path_str):
    if os.path.isdir(path_str):
        for f in ["pytorch_model.bin", "model.safetensors"]:
            p = os.path.join(path_str, f)
            if os.path.exists(p):
                return torch.load(p, map_location='cpu')
    return torch.load(path_str, map_location='cpu')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--model_paths", type=str, nargs="+", required=True)
    parser.add_argument("--tasks", type=str, nargs="+", required=True)
    parser.add_argument("--excel_path", type=str, required=True)
    parser.add_argument("--level", type=str, default="task_wise", choices=["task_wise", "layer_wise"])
    parser.add_argument("--search_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    task_specific_paths = {
        "cola": "CABS_Ex/models/textattack_roberta-base-CoLA",
        "sst2": "CABS_Ex/models/textattack_roberta-base-SST-2",
        "mrpc": "CABS_Ex/models/textattack_roberta-base-MRPC",
        "rte": "CABS_Ex/models/textattack_roberta-base-RTE",
        "stsb": "CABS_Ex/models/textattack_roberta-base-STS-B",
        "squad": "CABS_Ex/models/roberta-base-squad2",
        "race": "CABS_Ex/models/kda-roberta-base-race",
    }

    print("Preloading Heads for Search Phase...")
    heads_cache = {}
    for task in args.tasks:
        path = task_specific_paths.get(task.lower())
        if path:
            state = load_checkpoint_state_dict(path)

            if task.lower() == "squad":
                target = ["qa_outputs.weight", "qa_outputs.bias"]
            elif task.lower() == "race":
                target = [
                    "classifier.weight", "classifier.bias",
                    "classifier.dense.weight", "classifier.dense.bias",
                    "classifier.out_proj.weight", "classifier.out_proj.bias"
                ]
            else:
                target = ["classifier.dense.weight", "classifier.dense.bias", "classifier.out_proj.weight", "classifier.out_proj.bias"]

            head_sd = {}
            for k in state.keys():
                clean_k = k.replace("roberta.", "").replace("model.", "")
                if clean_k in target:
                    head_sd[clean_k] = state[k].to(device)

            if not head_sd:
                print(f"[Warning] No head weights loaded for {task}! Keys in checkpoint: {list(state.keys())[:5]}...")

            if head_sd:
                heads_cache[task.lower()] = head_sd

    tokenizer = RobertaTokenizerFast.from_pretrained(args.base_model_path)
    pool = FastGPUTaskVectorPool(args.base_model_path, args.tasks, args.model_paths, device=device)
    engine = AWA_Engine(pool, tokenizer, args.tasks, heads_cache, batch_size=args.batch_size)

    print("\n>>> Starting AWA Search...")
    best_coeffs = engine.run(mode=args.level, max_steps=args.search_steps)
    print(f"\n>>> Best Coefficients: {best_coeffs}")

    print("Applying best coefficients for Final Evaluation...")
    pool.apply_coeffs(best_coeffs, mode=args.level)
    final_state_dict = pool.base_model.state_dict()

    eval_sd = {}
    for k, v in final_state_dict.items():
        eval_sd[f"roberta.{k}"] = v

    print("\n>>> Final Evaluation (using evaluationpp.py)...")
    num_labels_map = {"cola": 2, "sst2": 2, "mrpc": 2, "rte": 2, "stsb": 1, "squad": 0, "race": 4, "mnli": 3, "qnli": 2}

    task_performance = {}
    for task in args.tasks:
        print(f"Evaluating {task}...")
        evaluator = TaskEvaluator(
            task=task,
            base_model_dir=args.base_model_path,
            task_specific_path=task_specific_paths.get(task.lower()),
            num_labels=num_labels_map.get(task.lower(), 2),
            tokenizer=tokenizer,
            device=device
        )
        score = evaluator.evaluate(eval_sd)
        task_performance[task] = score
        print(f"  -> Score: {score}")

    data = []
    header = ['algorithm', 'learned_weights']
    row = ["CABS+", json.dumps(best_coeffs.tolist() if isinstance(best_coeffs, np.ndarray) else best_coeffs)]

    avg_list = []
    first_pass = True
    for task in args.tasks:
        val = task_performance[task]
        if isinstance(val, dict):
            for k, v in val.items():
                if first_pass:
                    header.append(f"{task}_{k}")
                row.append(v)
            score = val.get('f1', val.get('exact_match', 0)) / 100.0
            avg_list.append(score)
        else:
            if first_pass:
                header.append(task)
            row.append(val)
            avg_list.append(val)

    if first_pass:
        header.append('average')
    row.append(sum(avg_list)/len(avg_list) if avg_list else 0)
    data.append(row)

    df = pd.DataFrame([row], columns=header)
    if os.path.dirname(args.excel_path):
        os.makedirs(os.path.dirname(args.excel_path), exist_ok=True)
    if args.excel_path.endswith('.xlsx'):
        df.to_excel(args.excel_path, index=False)
    else:
        df.to_csv(args.excel_path, index=False)
    print(f"Results saved to {args.excel_path}")
