# 7B Model Experiments Framework

This document describes the framework for conducting experiments on the 7B model using model pruning, merging, and evaluation methodologies. This framework extends similar approaches from the RoBERTa experiments to a larger-scale model.

## Overview

The 7B model experiments are divided into three main steps:

1. **Convert dtype**: convert the dtype of base model to float16 for avoiding error caused by dtype inconsistency.
2. **Extract Task Vector**: Extract task vector by subtracting the base model parameters from the fine-tuned model.
3. **Sparsification**: Apply various sparsification methods to the extracted task vectors.
4. **Merging and Evaluation**: Merge the pruned task vectors with the base model and evaluate on tasks.

## Models

we utilized pre-trained and fine-tuned versions of the Qwen2.5 model, obtained from Hugging Face. Specifically, the models used in our experiments were built upon the [qwen-2.5-7b-instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) backbone. Fine-tuned variants used include:

- [fq2.5-7b-it](https://huggingface.co/ehristoforu/fq2.5-7b-it-normalize_false)
- [Tsunami-0.5-7B-Instruct](https://huggingface.co/Tsunami-th/Tsunami-0.5-7B-Instruct)

## Files and Scripts

- **convert_dtype.py**: Code for convert model dtype
- **extract_task_vector_7b.py**: Code for extracting task vectors from the fine-tuned models by subtracting base model parameters.
- **prune_task_vector_7b.py**: Code for applying different sparsification methods to the task vectors.
- **main.py**：merging models using AWA strategy.

## Running Experiments

### 1. Convert dtype

convert the dtype of base model to float16 for avoiding error caused by dtype inconsistency.

```bash
python convert_dtype.py
```

### 2. Extract Task Vector

To extract task vectors, you will need to run the `extract_task_vector_7b.py` script twice with two different fine-tuned models, but the same base model.

#### First Run:

```bash
python extract_task_vector_7b.py \
    --finetuned_model_path path/to/fq2.5-7b-it \
    --base_model_path path/to/qwen-2.5-7b-instruct \
    --save_path /path/to/save/task_vector_fq2.5-7b-it
```

#### Second Run:

```bash
python extract_task_vector_7b.py \
    --finetuned_model_path path/to/Tsunami-0.5-7B-Instruct \
    --base_model_path path/to/qwen-2.5-7b-instruct \
    --save_path /path/to/save/task_vector_Tsunami-0.5-7B-Instruct
```

### 2. Sparsification of Task Vectors

Once the task vectors are extracted, apply sparsification to both vectors simultaneously using `prune_task_vector_7b.py`.

```bash
python prune_task_vector_7b.py \
    --task_vector_path1 /path/to/task_vector_fq2.5-7b-it \
    --task_vector_path2 /path/to/task_vector_Tsunami-0.5-7B-Instruct \
    --n 64 \
    --m 256 \
    --pruning_method "nm" \
    --save_directory1 /path/to/save/pruned_vector_fq2.5-7b-it \
    --save_directory2 /path/to/save/pruned_vector_Tsunami-0.5-7B-Instruct
```

### 3. Model Merging and Evaluation

Finally, merge the pruned task vectors with the base model and evaluate them using the provided recipes for MergeKit and then evaluate the merged model using lm-evaluation-harness.

- **MergeKit Recipe**: Use MergeKit to merge the pruned vectors.

Run MergeKit with the following command:
```
mergekit-yaml /src/mistrial_leaderboard_experiments/recipes/recipe.yml /path/to/save/models/model_name/
```

- **LM-Evaluation-Harness**: Use LM-Evaluation-Harness to evaluate the merged model on the specified tasks.

Example command:
```
lm-evaluation-harness --model hf --model_args pretrained=/path/to/models/model_name --tasks arc_challenge,hellaswag,truthfulqa_mc2,winogrande,gsm8k,mmlu --device cuda:0 --batch_size 8 --output_path results.json
```

## Hyperparameters

The hyperparameters for the 7B model experiments are controlled by the following arguments:

- **n**: Number of elements to keep in each group for pruning (required for `nm` pruning).
- **m**: Total number of elements in each group for pruning (required for `nm` pruning).
- **sparsity_level**: Defines the target sparsity level for pruning (required for `random` and `magnitude` pruning).
- **pruning_method**: Specifies the pruning method (`nm`, `random`, `magnitude`).

## Results

The results of the merging and evaluation step will be saved in **results.json**, containing metrics such as accuracy and task-specific performance.

## Notes

- Make sure to update paths to model files, task vectors, and results as required.
- Adjust hyperparameters to explore their impact on model performance.

