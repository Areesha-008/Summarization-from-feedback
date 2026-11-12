"""Task 3 - Stage 2: reward model.

RewardModel = SFT transformer body (model.model) + nn.Linear(hidden, 1) on the last
non-padding token. Bradley-Terry loss. 1 epoch, AdamW lr 1e-5, batch 8 pairs.
Report pairwise accuracy on 1,000 val pairs. Save to checkpoints/reward/.
"""
