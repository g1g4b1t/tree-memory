# Scaled Memory Benchmark Summary

## Dataset

- Concepts: 37
- Base facts: 111
- Updates: 17
- Queries: 256

## Overall

| memory | top1_correct | hit_at_k | retrieved_count | path_precision | wrong_path_hits | wrong_branch_hits | same_surface_wrong_path | stale_conflicts | context_contamination | ai_context_risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flat_append | 0.934 | 1.000 | 8.000 | 0.241 | 6.070 | 4.230 | 0.781 | 0.066 | 0.767 | 0.106 |
| flat_replace | 0.934 | 1.000 | 8.000 | 0.233 | 6.137 | 4.281 | 0.715 | 0.000 | 0.767 | 0.089 |
| hard_tree | 1.000 | 1.000 | 1.008 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hybrid_tree | 0.957 | 1.000 | 1.109 | 0.962 | 0.102 | 0.078 | 0.066 | 0.000 | 0.038 | 0.028 |

## By Task

| task | memory | top1_correct | hit_at_k | retrieved_count | path_precision | wrong_path_hits | wrong_branch_hits | same_surface_wrong_path | stale_conflicts | context_contamination | ai_context_risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conflict_isolation | flat_append | 0.000 | 1.000 | 8.000 | 0.125 | 7.000 | 6.118 | 2.000 | 0.000 | 0.875 | 0.250 |
| conflict_isolation | flat_replace | 0.000 | 1.000 | 8.000 | 0.125 | 7.000 | 6.118 | 1.000 | 0.000 | 0.875 | 0.125 |
| conflict_isolation | hard_tree | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| conflict_isolation | hybrid_tree | 0.353 | 1.000 | 2.529 | 0.422 | 1.529 | 1.176 | 1.000 | 0.000 | 0.578 | 0.422 |
| direct | flat_append | 1.000 | 1.000 | 8.000 | 0.245 | 6.036 | 4.135 | 1.252 | 0.000 | 0.755 | 0.157 |
| direct | flat_replace | 1.000 | 1.000 | 8.000 | 0.245 | 6.036 | 4.135 | 1.252 | 0.000 | 0.755 | 0.157 |
| direct | hard_tree | 1.000 | 1.000 | 1.009 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| direct | hybrid_tree | 1.000 | 1.000 | 1.009 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| local_update | flat_append | 1.000 | 1.000 | 8.000 | 0.250 | 6.000 | 4.353 | 0.000 | 1.000 | 0.875 | 0.125 |
| local_update | flat_replace | 1.000 | 1.000 | 8.000 | 0.125 | 7.000 | 5.118 | 0.000 | 0.000 | 0.875 | 0.000 |
| local_update | hard_tree | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| local_update | hybrid_tree | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| noisy_disambiguation | flat_append | 1.000 | 1.000 | 8.000 | 0.253 | 5.973 | 4.018 | 0.243 | 0.000 | 0.747 | 0.030 |
| noisy_disambiguation | flat_replace | 1.000 | 1.000 | 8.000 | 0.253 | 5.973 | 4.018 | 0.243 | 0.000 | 0.747 | 0.030 |
| noisy_disambiguation | hard_tree | 1.000 | 1.000 | 1.009 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| noisy_disambiguation | hybrid_tree | 1.000 | 1.000 | 1.009 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Checks

- hybrid_top1_close_to_strong_flat: True
- hybrid_path_precision_beats_flat: True
- hybrid_wrong_branches_below_flat: True
- hybrid_ai_context_risk_below_flat: True
- hybrid_stale_conflicts_below_append: True
- hybrid_beats_hard_on_hit_at_k: True
- final_pass: True