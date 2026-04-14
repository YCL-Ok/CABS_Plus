# CABS+: Efficient and Scalable Model Merging via Conflict-Aware Sparsification and Gradient-Free Weight Allocation

This repository contains the code for the submission "CABS+: Efficient and Scalable Model Merging via Conflict-Aware Sparsification and Gradient-Free Weight Allocation".

## Abstract

Model merging has recently attracted significant attention as an efficient and scalable paradigm for constructing unified multi-task models without requiring additional retraining. Among existing approaches, methods based on task vectors enable flexible integration of multiple tasks by combining expert models in the parameter space. However, due to the widespread presence of parameter conflicts and knowledge interference across tasks, the performance of merged models is often unsatisfactory.
To address these challenges, prior work introduced the **Conflict-Aware and Balanced Sparsification (CABS)** method, which reduces parameter interference through structured pruning and sequential masking. However, CABS relies on grid search to determine scaling coefficients, leading to the time complexity that grows exponentially with the number of tasks, thereby limiting its practical applicability. Meanwhile, its optimization objective is prone to being dominated by high performance tasks, resulting in imbalanced performance improvements across tasks and thus suboptimal overall performance. To overcome these limitations, we extend CABS and propose the enhanced method, termed **CABS+**. Specifically, the **Adaptive Weight Allocation (AWA)** strategy is proposed to efficiently optimize merging coefficients via the gradient-free search scheme. This design not only significantly reduces time complexity but also avoids substantial GPU memory overhead. Meanwhile, by incorporating **the asymmetric fitness function**, the proposed method effectively mitigates task bias during optimization, leading to more balanced performance gains across tasks. Moreover, to better understand the key factors influencing model merging performance, we conduct **the systematic empirical study** and propose the **Relative Synergy Score (RSS)** as the metric to quantify **model mergeability**, providing practical guidance for model selection in real-world applications. Extensive experimental results demonstrate that CABS+ achieves superior performance across various task configurations and model scales, while also offering notable advantages in memory efficiency and computational cost. Compared with existing methods, CABS+ exhibits stronger stability and robustness under varying task numbers and model architectures, and achieves better overall performance than state-of-the-art approaches.

## Summary figure

![image](https://github.com/user-attachments/assets/9ef9e2bf-d8b3-4a53-bbfc-6fb00f40dcf8)

Illustration of the overall framework of CABS+. (a) Conflict-Aware and Balanced Sparsification; (b) Adaptive Weight Allocation strategy; (c) Overall pipeline of CABS+.

## Code

### Install Dependencies

To set up the environment, use the provided `environment.yml` file:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate CABS
```

### Experiments

The repository contains two types of experiments:

1. **RoBERTa GLUE Experiments**: The experiments on RoBERTa are organized under the `src/roberta_glue_experiments` directory. These experiments include extracting task vectors, sparsifying them, merging, and evaluating the merged models on GLUE tasks.

2. **MIstral llm_leaderboard Experiments**: The experiments on the 7B parameter model are organized under the `src/mistral_leaderboard_experiments` directory. These experiments include using task vectors to perform model merging with MergeKit and evaluating with lm-evaluation-harness.

Detailed instructions for running the experiments are provided in each respective directory.
