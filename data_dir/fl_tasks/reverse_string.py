import torch

# Vocab: 0=PAD/DUMMY, 1=A, 2=B, 3=SEP (Separator/EOS)
vocab_size = 4 
PAD_IDX = 0
SEP_IDX = 3

def generate_sample(min_length, max_length, seed=None):
    """Generates a single sample for the Reverse String task."""
    
    # Set the seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    length = torch.randint(min_length, max_length + 1, (1,)).item()

    # Generate random binary string (tokens 1 and 2)
    sequence = [
        torch.randint(1, vocab_size - 1, (1,)).item() for _ in range(length)
    ]
    
    # Target is strictly the reverse of the sequence
    # (The model learns x -> x_reverse)
    target_sequence = sequence[::-1]

    # Return clean lists. 
    # We do NOT add [3] here, we add it as a separator in preprocess.
    return sequence, target_sequence


def preprocess_data(sample):
    """Preprocess function with Prompt + Dummy Token logic."""
    input_list, target_list = sample

    # 1. Calculate needed dummies
    # The paper requires appending |y| empty dummy tokens to the input.
    num_dummies = len(target_list)

    # 2. Construct Input: [String] + [SEP] + [DUMMY] * len(Result)
    # SEP (3) acts as the trigger to start processing dummies.
    # PAD (0) acts as the dummy token.
    input_seq = input_list + [SEP_IDX] + [PAD_IDX] * num_dummies
    
    # 3. Construct Target: [IGNORE] * len(Prompt) + [Result]
    # We ignore the loss for the prompt part ([String] + [SEP]).
    prompt_len = len(input_list) + 1 # +1 for SEP
    target_seq = [PAD_IDX] * prompt_len + target_list

    # 4. Construct Mask
    # 0 (False) for prompt, 1 (True) for result
    mask = torch.zeros(len(input_seq), dtype=torch.bool)
    mask[prompt_len:] = True

    return (
        torch.tensor(input_seq, dtype=torch.long),
        torch.tensor(target_seq, dtype=torch.long),
        mask
    )

# --- Quick Verification ---
# raw_sample = generate_sample(5, 5)
# print("Raw Input: ", raw_sample[0])
# print("Raw Target:", raw_sample[1])
# inp, tgt, msk = preprocess_data(raw_sample)
# print("Tensor Input:", inp.tolist())
# print("Tensor Target:", tgt.tolist())
# print("Mask:", msk.tolist())
