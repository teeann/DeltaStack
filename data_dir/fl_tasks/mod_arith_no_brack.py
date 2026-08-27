import torch

modulus = 5
vocab_size = modulus + 5


def generate_sample(min_length, max_length, seed=None):
    """Generates a single sample for the Modular Arithmetic task with Left-Associative Equal Precedence."""

    # Set the seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    original_length = torch.randint(min_length, max_length + 1, (1,)).item()

    # Ensure length is even to accommodate the "Number-Operator-Number...=" format
    if original_length % 2 == 1:
        length = original_length + 1
    else:
        length = original_length

    res = [None] * length

    # Fill in numbers (Tokens 5-9 represent digits 0-4)
    for i in range(0, length, 2):
        res[i] = torch.randint(5, 10, (1,)).item()

    # Fill in operators (Tokens 1-3 represent +, -, *)
    for i in range(1, length - 1, 2):
        res[i] = torch.randint(1, 4, (1,)).item()

    # Set the '=' operator at the last position
    res[-1] = 4

    # --- MODIFIED LOGIC START ---
    # Evaluate strictly left-to-right (Equal Precedence)

    current_val = res[0] - 5  # Initialize with the first number

    # Iterate through operator-number pairs
    for i in range(1, length - 1, 2):
        op = res[i]
        next_num = res[i + 1] - 5

        if op == 1:  # Addition ('+')
            current_val += next_num
        elif op == 2:  # Subtraction ('-')
            current_val -= next_num
        elif op == 3:  # Multiplication ('*') - processed immediately like + and -
            current_val *= next_num

    # --- MODIFIED LOGIC END ---

    # Calculate the final result modulo 5
    # Add 5 to map the result (0-4) back to token ID space (5-9)
    target = (current_val % 5) + 5

    return res, target


def preprocess_data(sample):
    """Preprocess function for the 'modular_arithmetic' task."""
    input, target = sample
    input_tensor = torch.tensor(input, dtype=torch.long)
    target_tensor = torch.zeros_like(input_tensor)
    target_tensor[-1] = target
    mask = torch.zeros(input_tensor.shape, dtype=torch.bool)
    mask[-1] = True
    return input_tensor, target_tensor, mask


def decode(sequence, target=None):

    # Define the vocabulary mapping for decoding
    vocab_map = {
        1: "+",
        2: "-",
        3: "*",
        4: "=",
    }
    # Add mappings for digits (Token 5 -> '0', Token 6 -> '1', etc.)
    for i in range(modulus):
        vocab_map[i + 5] = str(i)

    """
    Decodes the numeric sequence and target back into a readable string.
    
    Args:
        sequence (list or torch.Tensor): The input sequence of token IDs.
        target (int, optional): The target token ID.
    
    Returns:
        str: The decoded expression (e.g., "3+2*4=1").
    """
    # Decode the main sequence
    decoded_seq = "".join([vocab_map.get(token, "?") for token in sequence])

    # Append the target if provided
    if target is not None:
        decoded_target = vocab_map.get(target, "?")
        return f"{decoded_seq}{decoded_target}"

    return decoded_seq
