# RoBERTa GLUE Experiments

This repository contains the code and scripts for performing model merging, evaluation, and sparsification experiments using the RoBERTa model on GLUE tasks. The following sections provide an overview of how to use the scripts for different tasks.

## Overview

The experiments involve merging models, pruning model parameters, and evaluating their performance on various GLUE tasks. The main steps include:

1. **Extract Task Vector**: Extract task vectors by subtracting base model parameters from fine-tuned model parameters.
2. **Sparsification of Task Vectors**: Apply different sparsification methods to reduce the size of the extracted task vectors.
3. **Merging Models and Evaluation**: Merge the pruned task vectors with the base model and evaluate the resulting models on the GLUE benchmark to measure performance metrics.

**Models**: For each task, we utilized pre-trained and fine-tuned versions of RoBERTa, obtained from Hugging Face. Specifically, we used [FacebookAI/roberta-base](https://huggingface.co/facebook/roberta-base) as the base model. Fine-tuned models include [textattack/roberta-base-CoLA](https://huggingface.co/textattack/roberta-base-CoLA), [textattack/roberta-base-SST-2](https://huggingface.co/textattack/roberta-base-SST-2), [textattack/roberta-base-MRPC](https://huggingface.co/textattack/roberta-base-MRPC), and [textattack/roberta-base-RTE](https://huggingface.co/textattack/roberta-base-RTE).

## Files and Scripts

- **`main.py`**: Main code to merge models and evaluate them on GLUE tasks.
- **`evaluationpp.py`**: Evaluates merged models on specified GLUE tasks.
- **`preprocess.py`**: Preprocesses the dataset to prepare it for evaluation.
- **`extract_task_vector.py`**: Extracts the task vector by subtracting the base model's parameters from the fine-tuned model's parameters.
- **`prune_task_vector.py`**: Pruning task vectors using various sparsification methods, such as magnitude pruning, random pruning, and n pruning. 

## Running Experiments

### 1. Extract Task Vector

To extract the task vector from a fine-tuned model, run the `extract_task_vector.py` script with the required arguments:

```bash
python extract_task_vector.py \
    --finetuned_model_path /path/to/finetuned_model \
    --base_model_path /path/to/base_model \
    --save_path /path/to/save/task_vector
```

### 2. Sparsification of Task Vectors

To sparsify the extracted task vectors, use the `prune_task_vector.py` script with appropriate arguments to apply different pruning methods. The sparsification step can use magnitude pruning, random pruning, or n pruning. Example command:

For n pruning, specify the `n` and `m` values:

```bash
python prune_task_vector.py \
    --model1_path /path/to/task_vector1 \
    --model2_path /path/to/task_vector2 \
    --save_path1 /path/to/save/pruned_task_vector1 \
    --save_path2 /path/to/save/pruned_task_vector2 \
    --pruning_method n:m \
    --n 3 \
    --m 32
```

### 3. Model Merging and Evaluation

To merge and evaluate models, use the following command:

```bash
python main.py \
    --base_model_path models/FacebookAI_roberta-base \
    --model_paths \
        "reconstructed_tasksModel/four_CSRM/textattack_roberta-base-CoLA" \
        "reconstructed_tasksModel/four_CSRM/textattack_roberta-base-SST-2" \
        "reconstructed_tasksModel/four_CSRM/textattack_roberta-base-RTE" \
        "reconstructed_tasksModel/four_CSRM/textattack_roberta-base-MRPC" \
    --tasks cola sst2 rte mrpc \
    --excel_path "results.csv"
```

## Hyperparameters

The hyperparameters for model merging and evaluation are controlled by the following arguments:

- **`pruning_method`**: Specifies the pruning method (`magnitude`, `random`, `n:m`).
- **`sparsity_level`**: Specifies the target sparsity level for magnitude or random pruning.
- **`n`**, **`m`**: Parameters for n pruning, specifying how many elements to keep (`n`) in each group of size (`m`).
- **max_steps**: The maximum number of iterations for the AWA algorithm to run.
- **x0:** The initial starting point (center) for the coefficient search space. A value of 1.0 represents standard task vector addition.
- **bounds:** Defines the lower and upper limits for the coefficient search space to prevent extreme scaling.
- **sigma0:** The initial step size for the AWA search distribution.
- **popsize:** The number of candidate solutions generated and evaluated in each evolutionary generation.

## Results

The results of the model merging and evaluation are saved as a CSV file at the specified path (`results.csv`). This file contains the performance metrics for each task, including accuracy and average performance.

## Notes

- Ensure that the paths specified in the scripts (`model paths`, `save paths`, etc.) are updated to match your directory structure.
- Modify the hyperparameter values to explore different merging strategies and observe their impact on performance.

## License

This repository is licensed under the MIT License.
