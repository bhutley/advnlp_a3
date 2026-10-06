# FinBERT acceptance checks

## 1. Direction (must pass)

| model | set | correct | wrong |
|---|---|---|---|
| finbert_tone | positive | 4/4 | - |
| finbert_tone | negative | 4/4 | - |
| prosus_finbert | positive | 4/4 | - |
| prosus_finbert | negative | 4/4 | - |

## 2. MPS vs CPU (must pass, tolerance 0.01)

| model | device | n | max_emb_diff | max_prob_diff | pass |
|---|---|---|---|---|---|
| finbert_tone | mps | 200 | 4.77e-06 | 4.35e-06 | yes |
| prosus_finbert | mps | 200 | 5.72e-06 | 2.83e-06 | yes |

## 3. Cross-model agreement of call tone (reported only)

| n | pearson | spearman | spearman_p |
|---|---|---|---|
| 572 | 0.902 | 0.898 | 0.000 |

## 4. First look at the targets (reported only)

Spearman correlation over `ok` events. This is a sanity check, not a basis for choosing a model or pooling.

| model | feature | target | n | spearman | p |
|---|---|---|---|---|---|
| finbert_tone | tone | car_mm_0_1 | 490 | 0.187 | 0.000 |
| finbert_tone | tone | car_mm_0_5 | 490 | 0.219 | 0.000 |
| finbert_tone | tone | vol_3 | 490 | 0.059 | 0.194 |
| finbert_tone | tone | vol_30 | 490 | -0.021 | 0.647 |
| finbert_tone | frac_neg | car_mm_0_1 | 490 | -0.174 | 0.000 |
| finbert_tone | frac_neg | car_mm_0_5 | 490 | -0.206 | 0.000 |
| finbert_tone | frac_neg | vol_3 | 490 | 0.025 | 0.583 |
| finbert_tone | frac_neg | vol_30 | 490 | 0.051 | 0.261 |
| prosus_finbert | tone | car_mm_0_1 | 490 | 0.181 | 0.000 |
| prosus_finbert | tone | car_mm_0_5 | 490 | 0.218 | 0.000 |
| prosus_finbert | tone | vol_3 | 490 | 0.051 | 0.264 |
| prosus_finbert | tone | vol_30 | 490 | -0.039 | 0.387 |
| prosus_finbert | frac_neg | car_mm_0_1 | 490 | -0.147 | 0.001 |
| prosus_finbert | frac_neg | car_mm_0_5 | 490 | -0.192 | 0.000 |
| prosus_finbert | frac_neg | vol_3 | 490 | -0.019 | 0.668 |
| prosus_finbert | frac_neg | vol_30 | 490 | 0.025 | 0.577 |
