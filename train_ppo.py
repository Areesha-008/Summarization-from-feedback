"""Task 4 - Stage 3: PPO.

Policy + value head (from SFT), frozen reference (SFT), frozen reward model.
Rollout 16 prompts -> per-token KL-penalised reward -> returns/advantages (gamma=1)
-> K=4 PPO epochs (clipped policy loss + value loss + entropy bonus).
eps=0.2, beta_KL=0.05, c_V=0.1, c_H=0.01, AdamW lr 1e-5, clip 1.0, 200 iterations.
Save policy (no value head) to checkpoints/ppo/.
"""
