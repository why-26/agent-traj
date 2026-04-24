| Action | Model | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| Continue | SL Dual-Head | 0.9569 | 0.9711 | 0.9639 | 1555 |
| Continue | DT Dual-Head | 0.9473 | 0.9820 | 0.9643 | 1555 |
| Compress | SL Dual-Head | 0.8000 | 0.7899 | 0.7949 | 238 |
| Compress | DT Dual-Head | 0.8325 | 0.6891 | 0.7540 | 238 |
| Redirect | SL Dual-Head | 0.0000 | 0.0000 | 0.0000 | 1 |
| Redirect | DT Dual-Head | 0.0000 | 0.0000 | 0.0000 | 1 |
| ModeSwitch | SL Dual-Head | 0.8085 | 0.6667 | 0.7308 | 57 |
| ModeSwitch | DT Dual-Head | 0.6545 | 0.6316 | 0.6429 | 57 |
| Stop | SL Dual-Head | 0.8182 | 0.3913 | 0.5294 | 23 |
| Stop | DT Dual-Head | 0.2000 | 0.0870 | 0.1212 | 23 |

| Metric | SL Dual-Head | DT Dual-Head |
|---|---:|---:|
| Gate Precision | 0.8480 | 0.8931 |
| Gate Recall | 0.7868 | 0.7335 |