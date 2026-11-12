"""Task 5 - Stage 4: DPO.

Trainable policy + frozen reference, both from SFT. Sequence logprob = sum over summary
tokens. DPO loss with beta=0.1. 1 epoch, AdamW lr 2e-6, batch 8 pairs.
Log loss + implicit reward accuracy every 250 steps. Save to checkpoints/dpo/.
"""
