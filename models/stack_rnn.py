import torch
import torch.nn as nn
import torch.nn.functional as F


class StackRNNCell(nn.Module):
    """
    A single time-step of a Multi-Layer Stack RNN using nn.RNN.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        stack_cell_size,
        n_stacks=1,
        stack_size=20,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.stack_cell_size = stack_cell_size
        self.n_stacks = n_stacks
        self.stack_size = stack_size
        self.num_layers = num_layers

        # 1. Use standard nn.RNN with num_layers
        # Input to Layer 0 = (Original Input + Stack Read)
        # nn.RNN handles the internal transitions between layers automatically.
        self.rnn = nn.RNN(
            input_size=input_size + (n_stacks * stack_cell_size),
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Heads to predict stack actions from the LAST layer's hidden state
        self.push_val_head = nn.Linear(hidden_size, n_stacks * stack_cell_size)
        self.action_head = nn.Linear(hidden_size, n_stacks * 3)

    def _update_stack(self, stack, actions, push_value):
        """Standard differentiable stack update logic (Same as before)."""
        push_prob = actions[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        pop_prob = actions[:, :, 1].unsqueeze(-1).unsqueeze(-1)
        noop_prob = actions[:, :, 2].unsqueeze(-1).unsqueeze(-1)

        push_val_reshaped = push_value.unsqueeze(2)
        stack_shifted_down = torch.cat([push_val_reshaped, stack[:, :, :-1, :]], dim=2)

        zeros = torch.zeros_like(stack[:, :, 0:1, :])
        stack_shifted_up = torch.cat([stack[:, :, 1:, :], zeros], dim=2)

        return (
            (push_prob * stack_shifted_down)
            + (pop_prob * stack_shifted_up)
            + (noop_prob * stack)
        )

    def forward(self, input_step, prev_state):
        """
        input_step: (B, Input_Dim)
        prev_state: Tuple ( h, stack )
            h: (Num_Layers, B, Hidden)
        """
        h_prev, stack_prev = prev_state
        batch_size = input_step.size(0)

        # 1. Read Stack
        top_stacks = stack_prev[:, :, 0, :].reshape(batch_size, -1)

        # 2. Prepare Input: Concatenate Input + Stack
        # Shape: (Batch, Input_Dim + Stack_Width)
        rnn_input = torch.cat([input_step, top_stacks], dim=1)

        # 3. Step RNN
        # nn.RNN expects (Batch, Seq_Len, Dim). We create a fake seq_len of 1.
        rnn_input = rnn_input.unsqueeze(1)

        # output: (Batch, 1, Hidden) -> Output of the LAST layer
        # h_new: (Num_Layers, Batch, Hidden) -> States of ALL layers
        # Note: RNN does not maintain a cell state 'c' like LSTM
        output, h_new = self.rnn(rnn_input, h_prev)

        # Remove the sequence dimension
        final_hidden = output.squeeze(1)

        # 4. Compute Actions based on Top Layer (final_hidden)
        push_values = self.push_val_head(final_hidden).view(
            batch_size, self.n_stacks, self.stack_cell_size
        )
        action_probs = F.softmax(
            self.action_head(final_hidden).view(batch_size, self.n_stacks, 3), dim=-1
        )

        # 5. Update Stack
        stack_curr = self._update_stack(stack_prev, action_probs, push_values)

        return h_new, stack_curr


class StackRNN(nn.Module):
    def __init__(
        self,
        num_blocks,
        data_dim,
        model_dim,
        label_dim,
        dropout_rate,
        second_embedding=False,
        stack_size=20,
        n_stacks=1,
    ):
        super(StackRNN, self).__init__()

        self.second_embedding = second_embedding
        self.model_dim = model_dim
        self.num_blocks = num_blocks
        self.n_stacks = n_stacks
        self.stack_size = stack_size
        self.orig_stack_size = stack_size
        # self.stack_cell_size = model_dim
        self.stack_cell_size = 8

        embedding_dim = model_dim // 2 if second_embedding else model_dim
        self.embedding = nn.Embedding(data_dim, embedding_dim)

        # if second_embedding:
        #     self.embedding2 = nn.Embedding(data_dim, embedding_dim)

        self.cell = StackRNNCell(
            input_size=model_dim,
            hidden_size=model_dim,
            num_layers=num_blocks,  # Pass directly to RNN
            stack_cell_size=self.stack_cell_size,
            n_stacks=n_stacks,
            stack_size=stack_size,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(model_dim, label_dim)

    def set_stack_size(self, new_size: int):
        self.orig_stack_size = self.stack_size
        self.stack_size = new_size

    def reset_stack_size(self):
        self.stack_size = self.orig_stack_size

    def mask_grads(self):
        pass

    def forward(self, x):
        if not self.second_embedding:
            x = self.embedding(x)
        else:
            x = torch.cat(
                [self.embedding(x[:, :, 0]), self.embedding2(x[:, :, 1])], dim=-1
            )

        batch_size, seq_len, _ = x.size()

        # --- SIMPLIFIED STATE INITIALIZATION ---
        # nn.RNN expects states as single tensors of shape (Num_Layers, Batch, Hidden)
        # 'c' (cell state) initialization is removed as RNN doesn't use it.
        h = torch.zeros(self.num_blocks, batch_size, self.model_dim, device=x.device)

        stack = torch.zeros(
            batch_size,
            self.n_stacks,
            self.stack_size,
            self.stack_cell_size,
            device=x.device,
        )

        outputs = []

        for t in range(seq_len):
            input_t = x[:, t, :]
            input_t = self.dropout(input_t)

            # Step the cell
            # The state passed is simply (h, stack)
            h, stack = self.cell(input_t, (h, stack))

            # h contains states for all layers, we want the last layer for output
            outputs.append(h[-1])

        output_seq = torch.stack(outputs, dim=1)
        return self.linear(output_seq)
