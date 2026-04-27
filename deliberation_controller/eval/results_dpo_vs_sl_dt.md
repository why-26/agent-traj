# DPO vs SL vs DT

## Overall

| Model | Gate Acc | Action Acc | Overall Acc |
|---|---:|---:|---:|
| SL Dual-Head | 93.97% | 73.67% | 93.12% |
| DT | 93.97% | 63.32% | 92.26% |
| DPO | 90.55% | 40.75% | 89.49% |

## Per-Class P/R/F1

| Action | Model | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| Continue | SL | 0.9569 | 0.9711 | 0.9639 | 1555 |
| Continue | DT | 0.9473 | 0.9820 | 0.9643 | 1555 |
| Continue | DPO | 0.9015 | 0.9949 | 0.9459 | 1555 |
| Compress | SL | 0.8000 | 0.7899 | 0.7949 | 238 |
| Compress | DT | 0.8325 | 0.6891 | 0.7540 | 238 |
| Compress | DPO | 0.7956 | 0.4580 | 0.5813 | 238 |
| Redirect | SL | 0.0000 | 0.0000 | 0.0000 | 1 |
| Redirect | DT | 0.0000 | 0.0000 | 0.0000 | 1 |
| Redirect | DPO | 0.0000 | 0.0000 | 0.0000 | 1 |
| ModeSwitch | SL | 0.8085 | 0.6667 | 0.7308 | 57 |
| ModeSwitch | DT | 0.6545 | 0.6316 | 0.6429 | 57 |
| ModeSwitch | DPO | 1.0000 | 0.3509 | 0.5195 | 57 |
| Stop | SL | 0.8182 | 0.3913 | 0.5294 | 23 |
| Stop | DT | 0.2000 | 0.0870 | 0.1212 | 23 |
| Stop | DPO | 1.0000 | 0.0435 | 0.0833 | 23 |

## DPO vs SL Action-Level Delta (F1)

| Action | DPO F1 | SL F1 | Delta | Verdict |
|---|---:|---:|---:|---|
| Continue | 0.9459 | 0.9639 | -0.0180 | weaker |
| Compress | 0.5813 | 0.7949 | -0.2136 | weaker |
| Redirect | 0.0000 | 0.0000 | +0.0000 | same |
| ModeSwitch | 0.5195 | 0.7308 | -0.2113 | weaker |
| Stop | 0.0833 | 0.5294 | -0.4461 | weaker |
