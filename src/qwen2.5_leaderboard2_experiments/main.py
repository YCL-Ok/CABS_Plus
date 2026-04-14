import argparse
import os
import copy
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
import itertools
from tqdm import tqdm
import cma
import numpy as np
import gc


class HybridTaskVectorPool:
    """
    Manages base model on GPU and task vectors on CPU to prevent Out-Of-Memory (OOM) errors during evolutionary optimization.
    """
    def __init__(self, base_model_path, pruned_bin_paths, device="cuda"):
        self.device = device
        print(f"Loading Base Model from {base_model_path} directly to {device}...")
        
        # Load base model directly to GPU
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        self.base_model.eval()
        self.base_model.requires_grad_(False)
        self.base_state_dict = self.base_model.state_dict()
        
        self.num_layers = 0
        for key in self.base_state_dict.keys():
            match = re.search(r'\.(\d+)\.', key)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx > self.num_layers: self.num_layers = layer_idx
        self.num_layers += 1
        print(f"Detected {self.num_layers} layers.")
        
        # Load task specific deltas to CPU RAM
        self.task_deltas = []
        print("Loading Deltas to CPU RAM...")
        
        for p_path in pruned_bin_paths:
            print(f"  - Loading {os.path.basename(p_path)}...")
            loaded_sd = torch.load(p_path, map_location="cpu") 
            
            delta = {}
            for k, v in loaded_sd.items():
                if k in self.base_state_dict:
                    # Keep tensors on CPU in FP16 to minimize RAM usage
                    delta[k] = v.to(dtype=torch.float16) 
            
            self.task_deltas.append(delta)
            
        print(f"All Deltas successfully loaded to CPU.")

    def get_layer_index(self, key):
        match = re.search(r'\.(\d+)\.', key)
        return int(match.group(1)) if match else self.num_layers

    def apply_coeffs_and_forward(self, coeffs, batch, mode="task_wise"):
        """
        Dynamically applies coefficients to task vectors, performs a forward pass.
        """
        num_tasks = len(self.task_deltas)
        num_groups = self.num_layers + 1
        
        # Tracks modified parameters for subsequent restoration
        restore_info = [] 
        
        with torch.no_grad():
      
            all_keys = set()
            for d in self.task_deltas: all_keys.update(d.keys())
            
            for k in all_keys:
                total_delta_cpu = None
                
                for t_idx in range(num_tasks):
                    if k not in self.task_deltas[t_idx]: continue
                    
                    if mode == "task_wise":
                        c = coeffs[t_idx]
                    else:
                        l_idx = self.get_layer_index(k)
                        c = coeffs[t_idx * num_groups + l_idx]
                    
                    if abs(c) < 1e-6: continue
                    
                    delta_v = self.task_deltas[t_idx][k]
                    if total_delta_cpu is None:
                        total_delta_cpu = delta_v * c
                    else:
                        total_delta_cpu += delta_v * c
                
                if total_delta_cpu is not None:
                    param = self.base_state_dict[k]
                    
                    update_gpu = total_delta_cpu.to(self.device, non_blocking=True)
                    param.add_(update_gpu)
                    
                    restore_info.append((k, total_delta_cpu))
                    del update_gpu 

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            val_loss = loss.item()

            del outputs, shift_logits, shift_labels, loss

            for k, delta_cpu in restore_info:
                param = self.base_state_dict[k]
                update_gpu = delta_cpu.to(self.device, non_blocking=True)
                param.sub_(update_gpu)
                del update_gpu
            
            del restore_info
        
        return val_loss

class CalibrationManager:
    def __init__(self, tokenizer, batch_size=4): 
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.tokenizer.padding_side = "right"
        
        # Handle models lacking a default pad_token (e.g., Qwen, Llama)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")
            
    def get_loader(self):
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
            dataset = dataset.filter(lambda x: len(x['text']) > 100).select(range(64))
        except:
            print("Warning: WikiText load failed, utilizing dummy sentences.")
            from datasets import Dataset
            dataset = Dataset.from_dict({"text": ["This is a test sentence for calibration."] * 64})

        def tokenize_fn(examples):
            outputs = self.tokenizer(
                examples["text"], 
                truncation=True, 
                max_length=512, 
                padding="max_length", 
                return_tensors="pt"
            )
            
            labels = outputs["input_ids"].clone()
            
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            
            outputs["labels"] = labels
            return outputs

        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
        tokenized.set_format("torch")
        
        def tuple_collate_fn(features):
            batch = default_data_collator(features)
            return (batch,) 

        loader = DataLoader(tokenized, batch_size=self.batch_size, shuffle=True, collate_fn=tuple_collate_fn)
        return itertools.cycle(loader)

# =========================================================
# AWA Optimization Engine
# =========================================================
class AWA_Engine:
    def __init__(self, pool, tokenizer, batch_size):
        self.pool = pool
        self.data_manager = CalibrationManager(tokenizer, batch_size=batch_size)
        
    def run(self, mode="task_wise", max_steps=30):
        num_tasks = len(self.pool.task_deltas)
        if mode == "task_wise":
            num_params = num_tasks
        else:
            num_params = num_tasks * (self.pool.num_layers + 1)
            
        print(f"Optimizing {num_params} params. Mode: {mode}")

        x0 = [1.0] * num_params
        sigma0 = 0.05
        bounds = [0.10, 2.00]
        
        opts = {
            'popsize': 6, 
            'maxiter': max_steps, 
            'verbose': 1,
            'bounds': bounds
        }
        
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        
        best_loss_score = float('inf')
        best_coeffs = x0
        
        loader = self.data_manager.get_loader()
        fixed_batch = next(loader)[0]

        # Calculate baseline loss using standard vector addition (coeffs=1.0)
        print("Calculating Baseline Loss (coeffs=1.0)...")
        baseline_coeffs = [1.0] * num_params 
        baseline_loss = self.pool.apply_coeffs_and_forward(baseline_coeffs, fixed_batch, mode=mode)
        print(f"Baseline Loss: {baseline_loss:.4f}")
        
        pbar = tqdm(range(max_steps), desc="Evolving")
        for _ in pbar:
            if es.stop(): break
            
            solutions = es.ask()
            fitness_list = []
            
            for sol in solutions:
                # Apply boundary constraints
                sol = np.clip(sol, bounds[0], bounds[1])
                
                curr_loss = self.pool.apply_coeffs_and_forward(sol, fixed_batch, mode=mode)
                
                # Apply asymmetric penalty based on relative degradation against baseline
                relative_diff = (curr_loss - baseline_loss) / (baseline_loss + 1e-8)
                if relative_diff > 0:
                    loss_score = relative_diff * 100.0  # Heavy penalty for degradation
                else:
                    loss_score = relative_diff * 1.0    # Standard scaling for improvement
                
                fitness_list.append(loss_score)
                
            es.tell(solutions, fitness_list)
            es.disp()
            
            min_idx = np.argmin(fitness_list)
            current_best_score = fitness_list[min_idx]
            
            if current_best_score < best_loss_score:
                best_loss_score = current_best_score
                best_coeffs = solutions[min_idx]
            
            pbar.set_postfix({"BestScore": f"{best_loss_score:.4f}"})
            
            # Manual garbage collection to prevent VRAM fragmentation
            torch.cuda.empty_cache()
            
        return best_coeffs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolutionary Task Vector Optimization")
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model")
    parser.add_argument("--pruned_vectors", type=str, nargs="+", required=True, help="Paths to pruned task vectors")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged model")
    parser.add_argument("--level", type=str, default="task_wise", choices=["task_wise", "layer_wise"])
    parser.add_argument("--batch_size", type=int, default=8) 
    args = parser.parse_args()

    print("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    pool = HybridTaskVectorPool(args.base_model, args.pruned_vectors)
    engine = AWA_Engine(pool, tokenizer, batch_size=args.batch_size)
    
    best_coeffs = engine.run(mode=args.level, max_steps=50)
    
    print(f"Best Coefficients: {best_coeffs}")
    
    # Apply optimal coefficients and save the final merged model
    print("Applying optimal coefficients and saving to disk...")
    with torch.no_grad():
        num_tasks = len(pool.task_deltas)
        num_groups = pool.num_layers + 1
        for k in pool.base_state_dict.keys():
            total_delta_cpu = None
            for t_idx in range(num_tasks):
                if k in pool.task_deltas[t_idx]:
                    if args.level == "task_wise":
                        c = best_coeffs[t_idx]
                    else:
                        l_idx = pool.get_layer_index(k)
                        c = best_coeffs[t_idx * num_groups + l_idx]
                    
                    if abs(c) > 1e-6:
                        d = pool.task_deltas[t_idx][k]
                        if total_delta_cpu is None: total_delta_cpu = d * c
                        else: total_delta_cpu += d * c
            
            if total_delta_cpu is not None:
                pool.base_state_dict[k].add_(total_delta_cpu.to(pool.device))

    pool.base_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print("Done!")
