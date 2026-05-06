# Scaled Memory Benchmark Summary

## Dataset

- Concepts: 37
- Base facts: 111
- Updates: 17
- Queries: 256

## Overall

| memory | top1_correct | hit_at_k | retrieved_count | path_precision | wrong_path_hits | wrong_branch_hits | same_surface_wrong_path | stale_conflicts | context_contamination | ai_context_risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flat_append | 0.934 | 1.000 | 8.000 | 0.156 | 6.750 | 4.863 | 0.777 | 0.066 | 0.852 | 0.105 |
| flat_replace | 0.934 | 1.000 | 8.000 | 0.148 | 6.816 | 4.914 | 0.711 | 0.000 | 0.852 | 0.089 |
| gated_hybrid_tree | 1.000 | 1.000 | 1.059 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hard_tree | 1.000 | 1.000 | 1.059 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hybrid_tree | 0.957 | 1.000 | 1.160 | 0.962 | 0.102 | 0.078 | 0.066 | 0.000 | 0.038 | 0.028 |

## By Task

| task | memory | top1_correct | hit_at_k | retrieved_count | path_precision | wrong_path_hits | wrong_branch_hits | same_surface_wrong_path | stale_conflicts | context_contamination | ai_context_risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conflict_isolation | flat_append | 0.000 | 1.000 | 8.000 | 0.125 | 7.000 | 6.118 | 2.000 | 0.000 | 0.875 | 0.250 |
| conflict_isolation | flat_replace | 0.000 | 1.000 | 8.000 | 0.125 | 7.000 | 6.118 | 1.000 | 0.000 | 0.875 | 0.125 |
| conflict_isolation | gated_hybrid_tree | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| conflict_isolation | hard_tree | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| conflict_isolation | hybrid_tree | 0.353 | 1.000 | 2.529 | 0.422 | 1.529 | 1.176 | 1.000 | 0.000 | 0.578 | 0.422 |
| direct | flat_append | 1.000 | 1.000 | 8.000 | 0.149 | 6.811 | 4.883 | 1.243 | 0.000 | 0.851 | 0.155 |
| direct | flat_replace | 1.000 | 1.000 | 8.000 | 0.149 | 6.811 | 4.883 | 1.243 | 0.000 | 0.851 | 0.155 |
| direct | gated_hybrid_tree | 1.000 | 1.000 | 1.045 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| direct | hard_tree | 1.000 | 1.000 | 1.045 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| direct | hybrid_tree | 1.000 | 1.000 | 1.045 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| local_update | flat_append | 1.000 | 1.000 | 8.000 | 0.250 | 6.000 | 4.353 | 0.000 | 1.000 | 0.875 | 0.125 |
| local_update | flat_replace | 1.000 | 1.000 | 8.000 | 0.125 | 7.000 | 5.118 | 0.000 | 0.000 | 0.875 | 0.000 |
| local_update | gated_hybrid_tree | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| local_update | hard_tree | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| local_update | hybrid_tree | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| noisy_disambiguation | flat_append | 1.000 | 1.000 | 8.000 | 0.154 | 6.766 | 4.730 | 0.243 | 0.000 | 0.846 | 0.030 |
| noisy_disambiguation | flat_replace | 1.000 | 1.000 | 8.000 | 0.154 | 6.766 | 4.730 | 0.243 | 0.000 | 0.846 | 0.030 |
| noisy_disambiguation | gated_hybrid_tree | 1.000 | 1.000 | 1.090 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| noisy_disambiguation | hard_tree | 1.000 | 1.000 | 1.090 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| noisy_disambiguation | hybrid_tree | 1.000 | 1.000 | 1.090 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Checks

- hybrid_top1_close_to_strong_flat: True
- hybrid_path_precision_beats_flat: True
- hybrid_wrong_branches_below_flat: True
- hybrid_ai_context_risk_below_flat: True
- hybrid_stale_conflicts_below_append: True
- hybrid_beats_hard_on_hit_at_k: True
- gated_top1_close_to_hard: True
- gated_context_contamination_below_hybrid: True
- gated_ai_context_risk_below_hybrid: True
- gated_hit_at_k_ge_hybrid: True
- final_pass: True