import torch

num_elements = 2
# Vocab structure:
# 0: PAD
# 1..N: Stack Values (ST)
# N+1..2N: Push Values (PS)
# 2N+1: POP
# 2N+2: EMPTY (Optional, for empty stack result)
# 2N+3: ACT (Input delimiter)
# 2N+4: EOS (Termination)
vocab_size = 2 * num_elements + 4
# vocab_size = 2 * num_elements + 3
PAD_IDX = 0
ACT_IDX = num_elements * 2 + 3
# EOS_IDX = num_elements * 2 + 4


def generate_sample(min_length, max_length, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    # 1. Sample Lengths
    # Length acts as the "Total Input Budget"
    length = torch.randint(min_length, max_length + 1, (1,)).item()
    initial_stack_length = torch.randint(1, length, (1,)).item()

    # 2. Generate Data
    initial_stack = [
        torch.randint(1, num_elements + 1, (1,)).item()
        for _ in range(initial_stack_length)
    ]

    actions = [
        torch.randint(0, num_elements + 1, (1,)).item()
        for _ in range(length - initial_stack_length)
    ]

    # 3. Execute Stack Machine
    final_stack = initial_stack.copy()
    operations = []

    for action in actions:
        if action == 0:  # POP
            if final_stack:
                final_stack.pop()
            operations.append("POP")
        else:
            final_stack.append(action)
            operations.append(f"PS{action}")

    # 4. Return CLEAN lists (No padding here!)
    # Prompt: [ST... ops... ACT]
    prompt_list = [f"ST{el}" for el in initial_stack] + operations + ["ACT"]

    # Result: [ST... (reversed)]
    # Note: If stack is empty, we return empty list (or ["EMPTY"] if you prefer explicit token)
    if not final_stack:
        result_list = ["EMPTY"]
    else:
        result_list = [f"ST{el}" for el in final_stack[::-1]]

    return prompt_list, result_list


def preprocess_data(sample):
    prompt_list, result_list = sample

    # --- 1. Tokenization Helper ---
    def tokenize(token):
        if isinstance(token, int):
            return token  # Already mapped
        if token.startswith("ST"):
            return int(token[2:])
        if token.startswith("PS"):
            return int(token[2:]) + num_elements
        if token == "POP":
            return num_elements * 2 + 1
        if token == "EMPTY":
            return num_elements * 2 + 2
        if token == "ACT":
            return ACT_IDX
        return 0  # Default/Pad

    # Convert strings to integers
    prompt_ids = [tokenize(t) for t in prompt_list]
    result_ids = [tokenize(t) for t in result_list]

    # # OPTIONAL: Add EOS to result.
    # # Highly recommended so the model learns when to stop.
    # result_ids.append(EOS_IDX)

    # --- 2. The "Append Dummy" Logic ---
    num_dummies = len(result_ids)

    # Input: [Prompt] + [PAD] * len(Result)
    input_seq = prompt_ids + [PAD_IDX] * num_dummies

    # Target: [PAD] * len(Prompt) + [Result]
    target_seq = [PAD_IDX] * len(prompt_ids) + result_ids

    # Mask: 0 for prompt, 1 for result
    mask = torch.zeros(len(input_seq), dtype=torch.bool)
    mask[len(prompt_ids) :] = True

    return (
        torch.tensor(input_seq, dtype=torch.long),
        torch.tensor(target_seq, dtype=torch.long),
        mask,
    )
