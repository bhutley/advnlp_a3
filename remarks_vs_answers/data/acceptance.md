# Remarks vs answers: acceptance checks

## 1. Must pass

| check | pass |
|---|---|
| release matches saved outputs (build-time check) | yes |
| labels cover exactly the saved sentences | yes |
| sections in order in every call | yes |
| 488 modelling calls, each with prepared remarks | yes |
| calls.csv counts agree with sentences.csv | yes |

## 2. Rules used (modelling sample, reported)

| rule | calls |
|---|---|
| para1 | 425 |
| handover | 52 |
| handback | 11 |

## 3. Section sizes (modelling sample, reported)

| section | sentences | median per call |
|---|---|---|
| remarks | 34664 | 66.0 |
| answers | 43364 | 76.0 |
| closing | 435 | 0.0 |
| heading | 131 | 0.0 |

## 4. Calls with both sections (reported)

| min sentences in each | calls |
|---|---|
| 1 | 474 |
| 5 | 467 |
| 10 | 457 |
| 20 | 428 |

## 5. Answers' share of a call's remarks + answers (reported)

| quantile | share |
|---|---|
| 0.1 | 0.21 |
| 0.25 | 0.37 |
| 0.5 | 0.52 |
| 0.75 | 0.63 |
| 0.9 | 0.74 |

## 6. Hand audit (reported)

50 calls sampled. Wilson intervals.

| item | marked | correct | share | 95% CI |
|---|---|---|---|---|
| remarks/answers boundary | 50 | 50 | 1.00 | 0.93–1.00 |
| answers/final-paragraph boundary | 50 | 49 | 0.98 | 0.90–1.00 |

Boundary by rule:

| rule | marked | correct |
|---|---|---|
| handover | 5 | 5 |
| para1 | 45 | 45 |
