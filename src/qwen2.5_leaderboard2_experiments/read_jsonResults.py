import json
import sys
import numpy as np

def calculate_v2_score(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', {})
    
    # IFEval: average of prompt-level and instruction-level strict accuracy
    ifeval_score = 0
    if 'leaderboard_ifeval' in results:
        res = results['leaderboard_ifeval']
        p_strict = res.get('prompt_level_strict_acc,none', res.get('prompt_level_strict_acc', 0))
        i_strict = res.get('inst_level_strict_acc,none', res.get('inst_level_strict_acc', 0))
        ifeval_score = (p_strict + i_strict) / 2 * 100
    
    # BBH: mean normalized accuracy over all subtasks
    bbh_scores = []
    for k, v in results.items():
        if 'leaderboard_bbh' in k and 'alias' not in k:
            score = v.get('acc_norm,none', v.get('acc_norm', v.get('acc,none', v.get('acc', 0))))
            bbh_scores.append(score)
    bbh_score = np.mean(bbh_scores) * 100 if bbh_scores else 0
    
    # MATH (Hard): exact match
    math_scores = []
    for k, v in results.items():
        if 'leaderboard_math' in k and 'hard' in k:
            score = v.get('exact_match,none', v.get('exact_match', 0))
            math_scores.append(score)
    math_score = np.mean(math_scores) * 100 if math_scores else 0
    
    # GPQA: normalized accuracy
    gpqa_scores = []
    for k, v in results.items():
        if 'leaderboard_gpqa' in k:
            score = v.get('acc_norm,none', v.get('acc_norm', 0))
            gpqa_scores.append(score)
    gpqa_score = np.mean(gpqa_scores) * 100 if gpqa_scores else 0
    
    # MuSR: normalized accuracy
    musr_scores = []
    for k, v in results.items():
        if 'leaderboard_musr' in k:
            score = v.get('acc_norm,none', v.get('acc_norm', 0))
            musr_scores.append(score)
    musr_score = np.mean(musr_scores) * 100 if musr_scores else 0
    
    # MMLU-Pro: 5-shot accuracy
    mmlu_score = 0
    if 'leaderboard_mmlu_pro' in results:
        res = results['leaderboard_mmlu_pro']
        mmlu_score = res.get('acc,none', res.get('acc', 0)) * 100
    
    # Print results
    print(f"{'Task':<20} | {'Score':<10}")
    print("-" * 33)
    print(f"{'MMLU-Pro':<20} | {mmlu_score:.2f}")
    print(f"{'IFEval':<20} | {ifeval_score:.2f}")
    print(f"{'BBH':<20} | {bbh_score:.2f}")
    print(f"{'MATH (Hard)':<20} | {math_score:.2f}")
    print(f"{'GPQA':<20} | {gpqa_score:.2f}")
    print(f"{'MuSR':<20} | {musr_score:.2f}")
    print("-" * 33)
    
    final_avg = (mmlu_score + ifeval_score + bbh_score + math_score + gpqa_score + musr_score) / 6
    print(f"{'Average':<20} | {final_avg:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_jsonResults.py <path_to_json>")
    else:
        calculate_v2_score(sys.argv[1])
