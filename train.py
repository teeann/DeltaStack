import datetime
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import wandb
except Exception as _wandb_exc:  # noqa: BLE001
    # Logging is strictly optional -- results are parsed from the stdout logs --
    # so an unrelated broken dependency must not decide whether a run happens.
    print(f"WARNING: wandb unavailable ({type(_wandb_exc).__name__}: {_wandb_exc}); "
          f"continuing without experiment logging.")

    class _NoOpWandb:
        class _Config:
            def update(self, *args, **kwargs):
                pass

        config = _Config()

        def init(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            pass

        def finish(self, *args, **kwargs):
            pass

    wandb = _NoOpWandb()

# Local imports
from data_dir.dataloaders import (
    # create_a5_dataloaders,
    create_fl_dataloaders,
    # create_group_dataloaders,
)
from models.lr_scheduler import LinearWarmupCosineAnnealing


def set_seed(seed=42):
    """
    Sets the seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def vfA_l2(block):
    lcde = block.LCDE
    if hasattr(lcde, "vf_A"):
        tensors = [lcde.vf_A.weight if isinstance(lcde.vf_A, nn.Linear) else lcde.vf_A]
        for name in ("vf_A_u", "vf_A_v"):
            t = getattr(lcde, name, None)
            if t is not None:
                tensors.append(t)
    else:
        tensors = [lcde.vf_A_diag, lcde.vf_A_dense.weight]
    return sum((t**2).sum() ** 0.5 for t in tensors)


def set_model_stack_size(model: nn.Module, new_size: int):
    """
    Recursively finds all DeltaStack/StackRNN layers and updates their
    stack size.
    """
    from fla.layers import DeltaStack
    from models.stack_rnn import StackRNN

    count = 0
    for module in model.modules():
        if isinstance(module, (DeltaStack, StackRNN)):
            module.set_stack_size(new_size)
            count += 1
    print(f"Updated stack size to {new_size} for {count} layers.")


def reset_model_stack_size(model: nn.Module):
    """
    Recursively finds all DeltaStack/StackRNN layers and resets their
    stack size.
    """
    from fla.layers import DeltaStack
    from models.stack_rnn import StackRNN

    count = 0
    for module in model.modules():
        if isinstance(module, (DeltaStack, StackRNN)):
            module.reset_stack_size()
            count += 1
    print(f"Reset stack size for {count} layers.")


def train_model(
    config,
    data_dim,
    label_dim,
    task,
    model_name,
    model_dim,
    block_size,
    model,
    dataloader,
    num_steps,
    print_steps,
    learning_rate,
    device,
    early_stop_threshold=1.0,
    vf_A_norm_lambda=0.0,
    weight_decay_embedding=0.0,
    weight_decay_others=1e-2,
    warmup_fraction=0.15,
    final_lr=1e-5,
    run=0,
    val_stack_size=256,
):
    """
    Trains the model for a specified number of steps, logging progress, and optionally
    performing early stopping based on validation accuracy.

    Args:
        config (dict): Configuration parameters for training.
        data_dim (int): Dimensionality of the input data.
        label_dim (int): Dimensionality of the labels.
        task (str): Task name (e.g. A5).
        model_name (str): Name of the model.
        model (nn.Module): The model to be trained.
        dataloader (dict[str, DataLoader or Iterable]): Must contain "train" and "val" keys.
        num_steps (int): Total steps to train.
        print_steps (int): Frequency of logging and validation evaluation.
        learning_rate (float): Base learning rate for training.
        device (torch.device): Device for computation (CPU or GPU).
        early_stop_threshold (float, optional): Validation accuracy threshold for early stopping.
        weight_decay_embedding (float, optional): Weight decay for embedding parameters.
        weight_decay_others (float, optional): Weight decay for all other parameters.
        warmup_fraction (float, optional): Fraction of total steps for linear warmup.
        final_lr (float, optional): Final learning rate for cosine annealing.

    Returns:
        model (nn.Module): The trained model.
        steps (list[int]): Steps at which logs were printed.
        val_accs (list[float]): Validation accuracies per print step.
        early_stop (bool): Whether early stopping was triggered.
    """

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model.to(device)
    embedding_params = [p for n, p in model.named_parameters() if "embedding" in n]
    other_params = [p for n, p in model.named_parameters() if "embedding" not in n]

    optimizer = optim.AdamW(
        [
            {"params": embedding_params, "weight_decay": weight_decay_embedding},
            {"params": other_params, "weight_decay": weight_decay_others},
        ],
        lr=learning_rate,
    )

    warmup_steps = int(warmup_fraction * num_steps)
    lr_scheduler = LinearWarmupCosineAnnealing(
        optimizer, warmup_steps, num_steps, learning_rate, final_lr
    )

    criterion = nn.CrossEntropyLoss()
    step = 0
    total_loss = 0
    best_val_acc = 0
    total_train_acc = 0
    steps = []
    val_accs = []
    start_time = time.time()

    model.train()
    while step < num_steps:
        for X, y, mask in dataloader["train"]:
            optimizer.zero_grad()

            # Handle the case where we have multi-length data in a single batch
            if isinstance(X, tuple):
                X, X_2 = X
                y, y_2 = y
                mask, mask_2 = mask

                X_2, y_2 = X_2.to(device), y_2.to(device)
                outputs_2 = model(X_2)
                loss = criterion(outputs_2[mask_2], y_2[mask_2].flatten())

                loss.backward()

            X, y, mask = X.to(device), y.to(device), mask.to(device)
            outputs = model(X)
            # Norm
            norm = 0
            if hasattr(model, "blocks"):
                for block in model.blocks:
                    if hasattr(block, "LCDE"):
                        norm += vfA_l2(block)

            if task == "C4":
                pass
                # batch_size, seq_len, _ = outputs.shape
                # outputs.view(batch_size, seq_len, 42, 3)
                # outputs = outputs.reshape(-1, 3)
                # targets = y.reshape(-1)
                # loss = criterion(outputs, targets) + vf_A_norm_lambda * norm
            else:
                loss = (
                    criterion(outputs[mask], y[mask].flatten())
                    + vf_A_norm_lambda * norm
                )

                # Calculate Batch Accuracy (Standard)
                with torch.no_grad():
                    masked_outputs = outputs[mask]
                    masked_targets = y[mask].flatten()
                    if masked_targets.numel() > 0:
                        preds = masked_outputs.argmax(dim=-1)
                        batch_acc = (preds == masked_targets).float().mean().item()
                    else:
                        batch_acc = 0.0

            # Log norm to wandb if it's non-zero
            if norm > 0:
                wandb.log({"train/norm": norm.item()}, step=step)

            loss.backward()
            # Zero out any gradients if your model requires it
            model.mask_grads()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
            total_train_acc += batch_acc  # <--- Accumulate accuracy

            # Logging
            if step % print_steps == 0:
                set_model_stack_size(model, val_stack_size)
                model.eval()
                end_time = time.time()
                val_acc = 0
                val_num = 0

                with torch.no_grad():
                    for X_val, y_val, mask_val in dataloader["val"]:
                        X_val, y_val, mask_val = (
                            X_val.to(device),
                            y_val.to(device),
                            mask_val.to(device),
                        )
                        outputs_val = model(X_val)
                        if task == "C4":
                            batch_size, seq_len, _ = outputs_val.shape
                            outputs_val = outputs_val.view(batch_size, seq_len, 42, 3)
                            outputs_val = outputs_val.argmax(dim=-1)
                            correct = outputs_val == y_val
                            val_acc += correct.all(dim=-1).sum().item()
                            val_num += batch_size * seq_len
                        else:
                            val_acc += (
                                (
                                    outputs_val[mask_val].argmax(dim=-1)
                                    == y_val[mask_val].flatten()
                                )
                                .sum()
                                .item()
                            )
                            val_num += y_val[mask_val].flatten().size(0)

                if step == 0 and print_steps > 1:
                    total_loss *= print_steps
                    total_train_acc *= print_steps

                avg_loss = total_loss / print_steps
                accuracy = val_acc / val_num if val_num > 0 else 0
                avg_train_acc = total_train_acc / print_steps  # <--- Compute avg

                print(
                    f"Step: {step}, "
                    f"Loss: {avg_loss:.4f}, "
                    f"Train Acc: {avg_train_acc:.4f}, "  # <--- Print
                    f"Validation Acc: {accuracy:.4f}, "
                    f"Time: {end_time - start_time:.2f} sec"
                )

                # Log to wandb
                wandb.log(
                    {
                        "step": step,
                        "train/loss": avg_loss,
                        "train/accuracy": avg_train_acc,  # <--- Log
                        "val/accuracy": accuracy,
                        "val/best_accuracy": best_val_acc,
                        "time/elapsed": end_time - start_time,
                        "learning_rate": (
                            lr_scheduler.get_last_lr()[0]
                            if hasattr(lr_scheduler, "get_last_lr")
                            else learning_rate
                        ),
                    }
                )

                steps.append(step)
                val_accs.append(accuracy)

                timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                # Save model checkpoint
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "data_dim": data_dim,
                    "label_dim": label_dim,
                    "model_name": model_name,
                    "final_val_acc": val_accs[-1] if val_accs else None,
                    "timestamp": timestamp_str,
                }

                checkpoint_filename = f"checkpoint_{task}_{model_name}_{time_str}.pt"
                checkpoint_path = os.path.join("checkpoints", checkpoint_filename)
                if accuracy > best_val_acc:
                    best_val_acc = accuracy
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved model checkpoint to: {checkpoint_path}")

                early_stop = accuracy > early_stop_threshold

                out_filename = f"results_{task}_{model_name}_{time_str}.json"

                out_path = os.path.join("results", out_filename)

                # Gather all relevant info to save:
                results_dict = {
                    "config": config,
                    "steps": steps,
                    "val_accs": val_accs,
                    "early_stop": early_stop,
                }

                with open(out_path, "w") as f:
                    json.dump(results_dict, f, indent=2)

                print(f"Saved results to: {out_path}")

                start_time = time.time()
                total_loss = 0
                total_train_acc = 0

                reset_model_stack_size(model)
                model.train()

                if early_stop:
                    return model, steps, val_accs, True

            if step > num_steps:
                break

            step += 1

    return model, steps, val_accs, False


def run_experiment(config):
    """
    Main driver for the experiment. Creates dataloaders (either for A5 or formal language),
    creates the model, calls train_model(), and saves results to the results/ folder.
    """
    run = config.get("run", 0)
    seed = config.get("seed", 0) + run
    set_seed(seed)

    # Read config fields
    task = config["task"]  # e.g., "A5" or "majority"
    model_name = config["model_name"]  # e.g., "lcde" or "mamba"
    num_blocks = config["num_blocks"]
    num_heads = config.get("num_heads", 4)
    model_dim = config["model_dim"]
    batch_size = config["batch_size"]
    num_steps = config["num_steps"]
    print_steps = config["print_steps"]
    learning_rate = config["learning_rate"]
    wd = config.get("wd", 1e-2)

    # Initialize wandb
    wandb.init(
        project="formal-language-experiments",
        config=config,
        name=f"{config['task']}_{config['model_name']}_lr{learning_rate}",
        reinit=True,
    )

    # Optional fields
    diagonal = config.get("diagonal", False)
    diagonal_dense = config.get("diagonal_dense", False)
    fwht = config.get("fwht", False)
    use_glu = config.get("use_glu", False)
    second_embedding = config.get("second_embedding", False)
    gated = config.get("gated", True)
    early_stop_threshold = config.get("early_stop_threshold", 1.0)
    init_std = config.get("init_std", 1.0)
    block_size = config.get("block_size", 1)
    sparsity = config.get("sparsity", 1.0)
    dropout_rate = config.get("dropout_rate", 0.01)
    length = config.get("length")
    slstm_at = config.get("slstm_at", [1])
    vf_A_norm_lambda = config.get("vf_A_norm_lambda", 0.001)
    rank = config.get("rank", 0)

    stack_size = config.get("stack_size", None)
    # Eval-time stack size: evaluation runs at a larger stack than training
    # to test length generalization beyond the trained depth.
    val_stack_size = config.get("val_stack_size", 256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create DataLoader(s). Training samples short sequences and evaluation
    # samples longer ones, so validation accuracy measures length
    # generalization rather than in-distribution fit.
    train_padding_length = 128
    val_padding_length = 512 + 4  # for eos/separator

    train_dataloader, _, data_dim, label_dim = create_fl_dataloaders(
        task,
        num_samples=25600000,
        batch_size=batch_size,
        min_length=3,
        max_length=60,
        padding_length=train_padding_length,
        train_split=1.0,
        seed=seed,
    )
    val_dataloader, _, _, _ = create_fl_dataloaders(
        task,
        num_samples=8192,
        batch_size=batch_size,
        min_length=60,
        max_length=256,
        padding_length=val_padding_length,
        train_split=1.0,
        seed=2 * seed,
    )
    dataloader = {"train": train_dataloader, "val": val_dataloader}

    # Create the model
    if model_name == "mamba":
        from models.mamba import StackedMamba

        model = StackedMamba(
            num_blocks=num_blocks,
            model_dim=model_dim,
            data_dim=data_dim,
            label_dim=label_dim,
            dropout_rate=dropout_rate,
            use_glu=use_glu,
            second_embedding=second_embedding,
        )
    elif model_name == "S4":
        from models.S4 import StackedS4

        model = StackedS4(
            num_blocks=num_blocks,
            model_dim=model_dim,
            data_dim=data_dim,
            label_dim=label_dim,
            dropout_rate=dropout_rate,
            use_glu=use_glu,
            second_embedding=second_embedding,
        )
    elif model_name in ["deltanet", "gateddeltanet", "rwkv6", "deltaproduct"]:
        from models.fla import StackedBlock

        model = StackedBlock(
            layer_type=model_name,
            num_blocks=num_blocks,
            model_dim=model_dim,
            data_dim=data_dim,
            label_dim=label_dim,
            dropout_rate=dropout_rate,
            use_glu=use_glu,
            second_embedding=second_embedding,
            rank=rank,
            gated=gated,
        )
    elif model_name in ["deltastack"]:
        from models.fla import StackedBlock

        model = StackedBlock(
            num_heads=num_heads,
            layer_type=model_name,
            num_blocks=num_blocks,
            model_dim=model_dim,
            data_dim=data_dim,
            label_dim=label_dim,
            dropout_rate=dropout_rate,
            use_glu=use_glu,
            second_embedding=second_embedding,
            rank=rank,
            gated=gated,
            stack_size=stack_size,
        )
    elif model_name in ["deltanet2"]:
        from models.deltanet2 import StackedBlock

        model = StackedBlock(
            num_heads=num_heads,
            num_blocks=num_blocks,
            model_dim=model_dim,
            data_dim=data_dim,
            label_dim=label_dim,
            # sigmoid_scale=2,
            dropout_rate=dropout_rate,
            use_glu=use_glu,
            second_embedding=second_embedding,
        )

    elif model_name == "lcde":
        from models.slcde import StackedLCDE

        model = StackedLCDE(
            num_blocks=num_blocks,
            model_dim=model_dim,
            data_dim=data_dim,
            label_dim=label_dim,
            init_std=init_std,
            block_size=block_size,
            sparsity=sparsity,
            dropout_rate=dropout_rate,
            use_glu=use_glu,
            diagonal=diagonal,
            diagonal_dense=diagonal_dense,
            fwht=fwht,
            second_embedding=second_embedding,
            rank=rank,
        )
    elif model_name == "stack_rnn":
        from models.stack_rnn import StackRNN

        model = StackRNN(
            num_blocks=num_blocks,
            data_dim=data_dim,
            model_dim=model_dim,
            label_dim=label_dim,
            dropout_rate=dropout_rate,
            second_embedding=second_embedding,
            stack_size=stack_size,
        )

    elif model_name == "lstm":
        from models.lstm import LSTM

        model = LSTM(
            num_blocks=num_blocks,
            data_dim=data_dim,
            model_dim=model_dim,
            label_dim=label_dim,
            dropout_rate=dropout_rate,
            second_embedding=second_embedding,
        )
    elif model_name == "xlstm":
        from models.xlstm import xLSTM

        model = xLSTM(
            num_blocks=num_blocks,
            data_dim=data_dim,
            model_dim=model_dim,
            label_dim=label_dim,
            dropout_rate=dropout_rate,
            second_embedding=second_embedding,
            context_length=train_padding_length,
            slstm_at=slstm_at,
        )
    elif model_name == "transformer_causal":
        from models.transformer import CausalTransformer

        model = CausalTransformer(
            num_blocks=num_blocks,
            data_dim=data_dim,
            model_dim=model_dim,
            label_dim=label_dim,
            dropout_rate=dropout_rate,
            second_embedding=second_embedding,
        )
    else:
        raise ValueError("Model not recognized.")

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {pytorch_total_params}")
    wandb.config.update({"total_params": pytorch_total_params})

    # Create directories for results and checkpoints
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Train
    model, steps, val_accs, early_stop = train_model(
        config=config,
        data_dim=data_dim,
        label_dim=label_dim,
        task=task,
        model_name=model_name,
        model_dim=model_dim,
        block_size=block_size,
        model=model,
        dataloader=dataloader,
        num_steps=num_steps,
        print_steps=print_steps,
        learning_rate=learning_rate,
        early_stop_threshold=early_stop_threshold,
        # vf_A_norm_lambda=vf_A_norm_lambda,
        device=device,
        run=run,
        weight_decay_others=wd,
        val_stack_size=val_stack_size,
    )

    if early_stop:
        print("Early stop triggered! Finished before reaching num_steps.")
    else:
        print("Training complete. Did not trigger early stopping.")

    wandb.finish()
    return model, steps, val_accs, early_stop
