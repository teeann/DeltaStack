import torch

# Vocab: 0=PAD/DUMMY, 1=A, 2=B, 3=SEP (Separator/ACT)
vocab_size = 4
PAD_IDX = 0
SEP_IDX = 3


def generate_sample(min_length, max_length, seed=None):
    """
    Generates a single sample for the Copy String task.
    Example: [A, B, A] -> [A, B, A]
    """
    if seed is not None:
        torch.manual_seed(seed)

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    length = torch.randint(min_length, max_length + 1, (1,)).item()

    # Generate random binary string (tokens 1 and 2)
    sequence = [torch.randint(1, vocab_size - 1, (1,)).item() for _ in range(length)]

    # Target is an exact copy of the sequence
    target_sequence = list(sequence)

    return sequence, target_sequence


def preprocess_data(sample):
    """
    Preprocess function with Prompt + Dummy Token logic (Masked Prediction).
    Input:  [String] + [SEP] + [DUMMY] * len(Result)
    Target: [IGNORE] * len(Prompt) + [Result]
    """
    input_list, target_list = sample

    # 1. Calculate needed dummies
    num_dummies = len(target_list)

    # 2. Construct Input: [String] + [SEP] + [DUMMY] * len(Result)
    input_seq = input_list + [SEP_IDX] + [PAD_IDX] * num_dummies

    # 3. Construct Target: [IGNORE] * len(Prompt) + [Result]
    prompt_len = len(input_list) + 1  # +1 for SEP
    target_seq = [PAD_IDX] * prompt_len + target_list

    # 4. Construct Mask
    # 0 (False) for prompt, 1 (True) for result
    mask = torch.zeros(len(input_seq), dtype=torch.bool)
    mask[prompt_len:] = True

    return (
        torch.tensor(input_seq, dtype=torch.long),
        torch.tensor(target_seq, dtype=torch.long),
        mask,
    )


def decode_tokens(token_list):
    """Helper to visualize tokens as strings for debugging."""
    mapping = {0: "[DUMMY/IGNORE]", 1: "A", 2: "B", 3: "<SEP>"}
    return " ".join(mapping.get(t, "?") for t in token_list)


if __name__ == "__main__":
    print("=== Masked Prediction Copy String Debugging ===\n")

    for i in range(1):
        print(f"--- Sample {i+1} ---")

        # 1. Raw Generation
        raw_seq, raw_tgt = generate_sample(min_length=3, max_length=5)
        print(
            f"Human Readable: {decode_tokens(raw_seq)} \t -> \t {decode_tokens(raw_tgt)}"
        )

        # 2. Preprocessing for Masked Loss
        inp, tgt, msk = preprocess_data((raw_seq, raw_tgt))

        print("\nSynchronous Alignment:")
        print(f"Input (x):  {inp.tolist()}")
        print(f"Target (y): {tgt.tolist()}")
        print(f"Mask:       {[int(m) for m in msk.tolist()]}")

        # 3. Visualizing the Synchronous Prediction
        print("\nStep-by-step synchronous prediction:")
        inp_list = inp.tolist()
        tgt_list = tgt.tolist()
        for j in range(len(inp_list)):
            current_input_token = inp_list[j]
            token_it_predicts = tgt_list[j]
            is_calculated_in_loss = msk[j].item()

            calc_str = "-> [LOSS]" if is_calculated_in_loss else "-> [IGNORE]"
            print(
                f"  Input '{decode_tokens([current_input_token])}' \t predicts \t '{decode_tokens([token_it_predicts])}' \t {calc_str}"
            )
        print("\n" + "=" * 50 + "\n")
