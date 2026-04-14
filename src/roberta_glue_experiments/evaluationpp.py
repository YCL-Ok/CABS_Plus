import torch
import numpy as np
import evaluate
import os
import collections
from datasets import load_dataset
from transformers import (
    RobertaForSequenceClassification,
    RobertaForQuestionAnswering,
    RobertaForMultipleChoice,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from preprocess import load_and_cache_data

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Optional, Union


@dataclass
class DataCollatorForMultipleChoice:
    """Data collator for RACE multiple choice task."""
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


class TaskEvaluator:
    def __init__(self, task, base_model_dir, task_specific_path, num_labels, tokenizer, device):
        self.task = task.lower()
        self.device = device
        self.tokenizer = tokenizer
        self.num_labels = num_labels

        # Load dataset and preprocess
        if self.task == "squad":
            self.model_class = RobertaForQuestionAnswering
            raw_dataset = load_dataset("squad", split="validation")
            self.eval_examples = raw_dataset

            def prepare_squad(examples):
                questions = [q.strip() for q in examples["question"]]
                inputs = tokenizer(
                    questions, examples["context"],
                    max_length=384, truncation="only_second", stride=128,
                    return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length"
                )
                sample_map = inputs.pop("overflow_to_sample_mapping")
                example_ids = []
                for i in range(len(inputs["input_ids"])):
                    sample_idx = sample_map[i]
                    example_ids.append(examples["id"][sample_idx])
                    sequence_ids = inputs.sequence_ids(i)
                    offset = inputs["offset_mapping"][i]
                    inputs["offset_mapping"][i] = [o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)]
                inputs["example_id"] = example_ids
                return inputs

            self.eval_dataset = raw_dataset.map(prepare_squad, batched=True, remove_columns=raw_dataset.column_names)
            self.data_collator = DataCollatorWithPadding(tokenizer)
            self.metric = evaluate.load("squad")

        elif self.task == "race":
            self.model_class = RobertaForMultipleChoice
            try:
                raw_dataset = load_dataset("race", "all", split="validation")
            except:
                raw_dataset = load_dataset("race", "high", split="validation")

            def prepare_race(examples):
                first = [[c] * 4 for c in examples["article"]]
                second = [[f"{q} {o}" for o in opts] for q, opts in zip(examples["question"], examples["options"])]
                flat_first = sum(first, [])
                flat_second = sum(second, [])
                tokenized = tokenizer(flat_first, flat_second, truncation=True, max_length=512, padding="max_length")
                return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized.items()}

            label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
            raw_dataset = raw_dataset.map(lambda x: {"label": [label_map[a] for a in x["answer"]]}, batched=True)
            self.eval_dataset = raw_dataset.map(prepare_race, batched=True, remove_columns=[c for c in raw_dataset.column_names if c != "label"])
            try:
                self.data_collator = DataCollatorForMultipleChoice(tokenizer)
            except:
                self.data_collator = DataCollatorWithPadding(tokenizer)

        elif self.task == "stsb":
            self.model_class = RobertaForSequenceClassification
            self.num_labels = 1
            raw_datasets = load_and_cache_data("stsb", tokenizer)
            self.eval_dataset = raw_datasets["validation"]
            self.data_collator = DataCollatorWithPadding(tokenizer)

        else:
            self.model_class = RobertaForSequenceClassification
            raw_datasets = load_and_cache_data(task, tokenizer)
            self.eval_dataset = raw_datasets["validation"]
            self.data_collator = DataCollatorWithPadding(tokenizer)

        # Load base model
        if self.task in ["squad", "race"]:
            self.model = self.model_class.from_pretrained(base_model_dir).to(device)
        else:
            self.model = self.model_class.from_pretrained(base_model_dir, num_labels=self.num_labels).to(device)
        self.model.eval()

        # Load classification head weights
        self.classifier_weights = {}
        if task_specific_path:
            if os.path.isdir(task_specific_path):
                f_path = os.path.join(task_specific_path, "pytorch_model.bin")
            else:
                f_path = task_specific_path

            if os.path.exists(f_path):
                state = torch.load(f_path, map_location='cpu')

                if self.task == "squad":
                    target = ["qa_outputs.weight", "qa_outputs.bias"]
                elif self.task == "race":
                    target = ["classifier.weight", "classifier.bias"]
                else:
                    target = ["classifier.dense.weight", "classifier.dense.bias", "classifier.out_proj.weight", "classifier.out_proj.bias"]

                for key in target:
                    if key in state:
                        val = state[key]
                    elif f"roberta.{key}" in state:
                        val = state[f"roberta.{key}"]
                    else:
                        val = None
                    if val is not None:
                        self.classifier_weights[key] = val.to(device)
            else:
                print(f"Warning: Head path {f_path} not found.")

        # Trainer setup
        remove_cols = True
        batch_size = 64
        if self.task == "race":
            batch_size = 8
        if self.task == "squad":
            batch_size = 32
            remove_cols = False

        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=f"./tmp/{task}",
                per_device_eval_batch_size=batch_size,
                no_cuda=not torch.cuda.is_available()
            ),
            data_collator=self.data_collator
        )

    def postprocess_squad(self, predictions):
        start_logits, end_logits = predictions.predictions
        features_per_example = collections.defaultdict(list)
        for i, f in enumerate(self.eval_dataset):
            features_per_example[f["example_id"]].append(i)

        final_preds = {}
        for ex in self.eval_examples:
            valid_answers = []
            for idx in features_per_example[ex["id"]]:
                sl = start_logits[idx]
                el = end_logits[idx]
                offset = self.eval_dataset[idx]["offset_mapping"]

                s_idx = np.argsort(sl)[-1]
                e_idx = np.argsort(el)[-1]

                if s_idx >= len(offset) or e_idx >= len(offset) or offset[s_idx] is None or offset[e_idx] is None or e_idx < s_idx:
                    continue
                valid_answers.append({
                    "score": sl[s_idx] + el[e_idx],
                    "text": ex["context"][offset[s_idx][0]:offset[e_idx][1]]
                })

            if valid_answers:
                final_preds[ex["id"]] = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]["text"]
            else:
                final_preds[ex["id"]] = ""

        formatted = [{"id": k, "prediction_text": v} for k, v in final_preds.items()]
        refs = [{"id": ex["id"], "answers": ex["answers"]} for ex in self.eval_examples]
        return self.metric.compute(predictions=formatted, references=refs)

    def evaluate(self, merged_state):
        # Load merged backbone weights
        self.model.load_state_dict(merged_state, strict=False)

        # Load classification head if provided
        if self.classifier_weights:
            with torch.no_grad():
                if self.task == "squad":
                    self.model.qa_outputs.weight.copy_(self.classifier_weights.get('qa_outputs.weight', self.model.qa_outputs.weight))
                    self.model.qa_outputs.bias.copy_(self.classifier_weights.get('qa_outputs.bias', self.model.qa_outputs.bias))
                elif self.task == "race":
                    self.model.classifier.weight.copy_(self.classifier_weights.get('classifier.weight', self.model.classifier.weight))
                    self.model.classifier.bias.copy_(self.classifier_weights.get('classifier.bias', self.model.classifier.bias))
                elif hasattr(self.model, 'classifier'):
                    cw = self.classifier_weights
                    self.model.classifier.dense.weight.copy_(cw.get('classifier.dense.weight', self.model.classifier.dense.weight))
                    self.model.classifier.dense.bias.copy_(cw.get('classifier.dense.bias', self.model.classifier.dense.bias))
                    self.model.classifier.out_proj.weight.copy_(cw.get('classifier.out_proj.weight', self.model.classifier.out_proj.weight))
                    self.model.classifier.out_proj.bias.copy_(cw.get('classifier.out_proj.bias', self.model.classifier.out_proj.bias))

        output = self.trainer.predict(self.eval_dataset)

        if self.task == "stsb":
            predictions = np.squeeze(output.predictions)
            return (predictions == output.label_ids).mean()
        elif self.task == "squad":
            return self.postprocess_squad(output)
        elif self.task == "race":
            predictions = np.argmax(output.predictions, axis=1)
            return (predictions == output.label_ids).mean()
        else:
            predictions = np.argmax(output.predictions, axis=1)
            return (predictions == output.label_ids).mean()
