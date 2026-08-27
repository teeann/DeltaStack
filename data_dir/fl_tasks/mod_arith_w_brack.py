import torch

# Configuration
modulus = 5
# Size = Padding(1) + Digits(5) + Ops/Brackets(7) = 13
vocab_size = modulus + 8


def decode(sample, modulus=5):
    """
    Decodes the output of generate_sample back into a readable string.

    Args:
        sample: A tuple (input_sequence, target_value)
        modulus: The modulus used in generation (default 5)

    Returns:
        A string representation like "(1+2)=3"
    """
    input_seq, target_val = sample

    # Create reverse mapping (Index -> Char)
    vocab = {
        0: "<PAD>",
        modulus + 1: "+",
        modulus + 2: "-",
        modulus + 3: "*",
        modulus + 4: "(",
        modulus + 5: ")",
        # modulus + 6 is reserved for 'x' (unused in standard generation)
        modulus + 6: "=",
    }

    # Add digits 0..modulus-1 (mapped to indices 1..modulus)
    for i in range(modulus):
        vocab[i + 1] = str(i)

    # Decode input sequence
    decoded_expression = []
    for idx in input_seq:
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        decoded_expression.append(vocab.get(idx, "?"))

    # Decode target
    if isinstance(target_val, torch.Tensor):
        target_val = target_val.item()
    decoded_target = vocab.get(target_val, "?")

    # Combine: The input sequence already contains the '=' at the end
    return "".join(decoded_expression) + decoded_target


def generate_sample(min_length, max_length, seed=None, mult=True):
    """
    Generates a sample for Modular Arithmetic with brackets.
    Logic matches DeepMind JAX implementation, but indices are shifted by +1
    to reserve index 0 for padding.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    # JAX check: "Can't generate expressions of length < 1"
    if min_length < 1:
        raise ValueError(f"Can't generate expressions of length < 1. Got {min_length}.")

    def gen_terminal():
        """Generates a random terminal value."""
        # Logic matches JAX: np.random.randint(low=0, high=modulus)
        value = torch.randint(0, modulus, (1,)).item()
        return str(value), value

    def generate_expression(length):
        """Recursively generates an expression of the given length."""
        if length < 1:
            raise ValueError(f"Can't generate expressions of length < 1. Got {length}.")

        # Base cases (Lengths 1-4) matching JAX
        if length == 1:
            return gen_terminal()
        elif length == 2:
            term_str, term_val = gen_terminal()
            return f"-{term_str}", -term_val % modulus
        elif length == 3:
            term_str, term_val = gen_terminal()
            return f"({term_str})", term_val % modulus
        elif length == 4:
            term_str, term_val = gen_terminal()
            return f"(-{term_str})", -term_val % modulus

        # Recursive step for length > 4
        # JAX: left_length = np.random.randint(low=1, high=length - 3)
        left_length = torch.randint(1, length - 3, (1,)).item()
        right_length = length - (left_length + 3)

        left_str, left_val = generate_expression(left_length)
        right_str, right_val = generate_expression(right_length)

        # Sample operator
        # JAX: maxop = 3 if mult else 2; op = randint(0, maxop)
        max_op = 3 if mult else 2
        op = torch.randint(0, max_op, (1,)).item()

        if op == 0:  # +
            return (
                "(" + left_str + "+" + right_str + ")",
                (left_val + right_val) % modulus,
            )
        elif op == 1:  # -
            return (
                "(" + left_str + "-" + right_str + ")",
                (left_val - right_val) % modulus,
            )
        else:  # * (op == 2)
            return (
                "(" + left_str + "*" + right_str + ")",
                (left_val * right_val) % modulus,
            )

    # Generate an expression of random length
    length = torch.randint(min_length, max_length + 1, (1,)).item()
    expression_str, result = generate_expression(length)

    # --- Vocabulary Mapping (Shifted +1 for Padding) ---
    # 0 is Padding
    # 1..mod are Digits 0..mod-1
    # mod+1..mod+7 are Ops

    vocab = {
        "+": modulus + 1,
        "-": modulus + 2,
        "*": modulus + 3,
        "(": modulus + 4,
        ")": modulus + 5,
        # 'x': modulus + 6, # Reserved place
        "=": modulus + 7,
    }
    for i in range(modulus):
        vocab[str(i)] = i + 1

    input_sequence = [vocab[char] for char in expression_str]
    input_sequence.append(vocab["="])

    target_value = vocab[str(result)]

    return input_sequence, target_value


def preprocess_data(sample):
    """
    Preprocess function remains the same.
    """
    input_sequence, target_value = sample

    input_tensor = torch.tensor(input_sequence, dtype=torch.long)
    target_tensor = torch.zeros_like(input_tensor, dtype=torch.long)
    target_tensor[-1] = target_value

    mask = torch.zeros(input_tensor.shape, dtype=torch.bool)
    mask[-1] = True

    return input_tensor, target_tensor, mask
