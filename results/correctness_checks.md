# Correctness checks (Section 3.3)

- **SFT mask** (PASS): decoding the tokens where summary_mask = 1 gives exactly the summary + EOS, e.g. `' Me and my sister were going to the park, put my sister in her carseat, left her there for thirty minutes, mom saved her from my ignorance, got grounded for a month.<|endoftext|>'`.
- **Frozen models** (PASS): every parameter of pi_ref and r_phi has requires_grad = False in Stage 3 (ref True, reward True) and of pi_ref in Stage 4 (True).
- **PPO, first iteration** (PASS): mean KL to the reference = 0.00e+00.
- **PPO, every iteration** (PASS): in the first of the K epochs, max |r_t(theta) - 1| over all iterations = 0.00e+00, max clip fraction = 0.0.
- **DPO, first step** (PASS): before any update the implicit reward accuracy on the 1,000 validation pairs is 0.000, i.e. every margin beta*(log-ratio_w - log-ratio_l) is exactly 0 because pi_theta = pi_ref, so the loss is -log sigma(0) = log 2 = 0.6931. (The printed first-step loss was not kept because that DPO run resumed from a save made before the value was recorded.)
- **Padding**: prompts for `generate` are left-padded (`collate_prompts`); the reward model reads the hidden state at index `attention_mask.sum(1) - 1` of right-padded rows, i.e. the last real token. Real tokens are always found with `attention_mask`, never with `input_ids != pad_token_id`.
