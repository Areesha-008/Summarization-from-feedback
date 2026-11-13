# Table 2 - Training

| Stage | Final train loss | Validation metric | Time / GPU |
|---|---|---|---|
| SFT | 2.3391 | val. loss: 2.3508 | 10.8 min / NVIDIA A100-SXM4-80GB |
| Reward model | 0.6018 | pairwise acc.: 0.6420 | 20.6 min / NVIDIA A100-SXM4-80GB |
| PPO | -0.0110 | final mean r_phi: 2.2728 (last-10 mean 1.9884) | 14.3 min / NVIDIA A100-SXM4-80GB |
| DPO | 0.6181 | implicit reward acc.: 0.6100 | 34.6 min / NVIDIA A100-SXM4-80GB |
