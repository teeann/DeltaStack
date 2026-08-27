import torch

# Configuration for Dyck-2 (2 pairs of brackets)
# Pairs: ( ) and [ ]
# Vocab: 0=PAD, 1=SEP, 2=(, 3=), 4=[, 5=]
BRACKET_PAIRS = [("(", ")"), ("[", "]")]
OPENERS = ["(", "["]
CLOSERS = [")", "]"]
PAIR_MAP = {"(": ")", "[": "]"}

# 0=PAD, 1=SEP, 2..5 = Brackets
vocab_size = 6
PAD_IDX = 0
SEP_IDX = 1

# Reverse lookup for decoding
ID_TO_CHAR = {0: "<PAD>", 1: "<SEP>", 2: "(", 3: ")", 4: "[", 5: "]"}
# Forward lookup for generation
CHAR_TO_ID = {v: k for k, v in ID_TO_CHAR.items()}


def decode(sample):
    """
    Decodes the output sample into readable strings.
    Args:
        sample: A tuple (input_tensor, target_tensor) or (input_list, target_list)
    """
    input_seq, target_seq = sample

    def seq_to_str(seq):
        chars = []
        for idx in seq:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            chars.append(ID_TO_CHAR.get(idx, "?"))
        return " ".join(chars)

    return f"Input:  {seq_to_str(input_seq)} - Target: {seq_to_str(target_seq)}"


# def generate_sample(min_length, max_length, seed=None):
#     """
#     Generates a valid Dyck prefix and the corresponding completion suffix.
#     """
#     if seed is not None:
#         torch.manual_seed(seed)

#     if min_length > max_length:
#         raise ValueError("min_length must be less than or equal to max_length")

#     length = torch.randint(min_length, max_length + 1, (1,)).item()

#     stack = []
#     input_sequence_chars = []

#     # NEW: Track depth history for the prefix
#     prefix_depths = []

#     prob_open = 0.8

#     for _ in range(length):
#         # Decision: 0 = Push (Open), 1 = Pop (Close)
#         # We bias towards Push (0.6) to ensure we build some stack depth to close later.
#         # If stack is empty, we MUST push.
#         if len(stack) > 0:
#             action = 1 if torch.rand(1).item() > prob_open else 0
#         else:
#             action = 0

#         if action == 0:  # PUSH
#             # Randomly pick '(' or '['
#             type_idx = torch.randint(0, len(OPENERS), (1,)).item()
#             char = OPENERS[type_idx]
#             stack.append(char)
#             input_sequence_chars.append(char)
#         else:  # POP
#             # To be a valid prefix, we must strictly close the last opener
#             opener = stack.pop()
#             closer = PAIR_MAP[opener]
#             input_sequence_chars.append(closer)

#         # NEW: Record depth after the operation
#         prefix_depths.append(len(stack))

#     # The target is the sequence of closers for whatever is left on the stack.
#     # If stack is ['(', '['], we need to output ']' then ')'
#     target_sequence_chars = [PAIR_MAP[op] for op in reversed(stack)]

#     # NEW: Calculate depth history for the suffix (it effectively just counts down)
#     # If stack depth is 3, suffix will close it: 2, 1, 0.
#     current_depth = len(stack)
#     suffix_depths = []
#     for _ in range(current_depth):
#         current_depth -= 1
#         suffix_depths.append(current_depth)

#     # Convert to indices
#     input_ids = [CHAR_TO_ID[c] for c in input_sequence_chars]
#     target_ids = [CHAR_TO_ID[c] for c in target_sequence_chars]

#     return input_ids, target_ids, prefix_depths, suffix_depths

def generate_sample(min_length, max_length, seed=None):
    """
    Generates a valid Dyck prefix and the corresponding completion suffix.
    Uses Heavy-Tailed Length Sampling and Markovian Depth Momentum.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    # --- Heavy-Tailed Length Sampling ---
    r = torch.rand(1).item()
    length = min_length + int((max_length - min_length + 1) * (r ** 0.5))
    length = max(min_length, min(length, max_length))

    stack = []
    input_sequence_chars = []
    prefix_depths = []
    
    # --- NEW: Markovian Depth Momentum ---
    # 0 = Pushing Phase, 1 = Popping Phase
    current_phase = 0 
    
    for _ in range(length):
        # If the stack is empty, we physically MUST push.
        # We also reset the momentum to the Pushing phase.
        if len(stack) == 0:
            action = 0
            current_phase = 0  
        else:
            # 85% chance to maintain the current momentum (create a burst).
            # 15% chance to switch phases (peak or valley).
            if torch.rand(1).item() > 0.85:
                current_phase = 1 - current_phase
            
            action = current_phase

        if action == 0:  # PUSH
            type_idx = torch.randint(0, len(OPENERS), (1,)).item()
            char = OPENERS[type_idx]
            stack.append(char)
            input_sequence_chars.append(char)
        else:  # POP
            opener = stack.pop()
            closer = PAIR_MAP[opener]
            input_sequence_chars.append(closer)

        prefix_depths.append(len(stack))

    # The target is the sequence of closers for whatever is left on the stack.
    target_sequence_chars = [PAIR_MAP[op] for op in reversed(stack)]

    # Calculate depth history for the suffix
    current_depth = len(stack)
    suffix_depths = []
    for _ in range(current_depth):
        current_depth -= 1
        suffix_depths.append(current_depth)

    # Convert to indices
    input_ids = [CHAR_TO_ID[c] for c in input_sequence_chars]
    target_ids = [CHAR_TO_ID[c] for c in target_sequence_chars]

    return input_ids, target_ids, prefix_depths, suffix_depths


def preprocess_data(sample):
    """Preprocess function now constructs the full depth tensor."""
    # NEW: Unpack 4 items
    input_list, target_list, prefix_depths, suffix_depths = sample

    # 1. Calculate needed dummies
    num_dummies = len(target_list)

    # 2. Construct Input: [Prefix] + [SEP] + [DUMMY] * len(Suffix)
    input_seq = input_list + [SEP_IDX] + [PAD_IDX] * num_dummies

    # 3. Construct Target
    prompt_len = len(input_list) + 1  # +1 for SEP
    target_seq = [PAD_IDX] * prompt_len + target_list

    # 4. Construct Mask
    mask = torch.zeros(len(input_seq), dtype=torch.bool)
    mask[prompt_len:] = True

    # 5. NEW: Construct Depth Tensor
    # Depth matches Input alignment:
    # [Prefix Depths] + [0 (for SEP)] + [Suffix Depths]
    # depth_seq = prefix_depths + [0] + suffix_depths

    return (
        torch.tensor(input_seq, dtype=torch.long),
        torch.tensor(target_seq, dtype=torch.long),
        mask,
        # torch.tensor(depth_seq, dtype=torch.long),  # New Return
    )



if __name__ == "__main__":
    print("=== Masked Prediction Dyck-2 Debugging ===\n")

    for i in range(2):
        print(f"--- Sample {i+1} ---")

        # 1. Raw Generation
        raw_input_ids, raw_target_ids, p_depths, s_depths = generate_sample(min_length=4, max_length=6)
        
        input_str = " ".join([ID_TO_CHAR[idx] for idx in raw_input_ids])
        target_str = " ".join([ID_TO_CHAR[idx] for idx in raw_target_ids])
        
        print(f"Prefix:  {input_str}")
        print(f"Suffix:  {target_str}")

        # 2. Preprocessing for Masked Loss
        sample_tuple = (raw_input_ids, raw_target_ids, p_depths, s_depths)
        inp, tgt, msk = preprocess_data(sample_tuple)

        print("\nSynchronous Alignment:")
        print(f"Input (x):  {inp.tolist()}")
        print(f"Target (y): {tgt.tolist()}")
        print(f"Mask:       {[int(m) for m in msk.tolist()]}")
        # print(f"Depth:      {depth.tolist()}")

        # 3. Visualizing the Synchronous Prediction
        print("\nStep-by-step synchronous prediction:")
        inp_list = inp.tolist()
        tgt_list = tgt.tolist()
        # depth_list = depth.tolist()
        
        for j in range(len(inp_list)):
            current_input_token = inp_list[j]
            token_it_predicts = tgt_list[j]
            # current_depth = depth_list[j]
            is_calculated_in_loss = msk[j].item()

            calc_str = "-> [LOSS]" if is_calculated_in_loss else "-> [IGNORE]"
            print(
                f"  In '{ID_TO_CHAR[current_input_token]:>5}' \t predicts \t "
                f"'{ID_TO_CHAR[token_it_predicts]:>5}' \t {calc_str}"
            )
        print("\n" + "=" * 60 + "\n")
