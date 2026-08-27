import argparse
import json
import os
import multiprocessing as mp

from train import run_experiment


def worker_run(gpu_id, config):
    """Worker process that masks all GPUs except the assigned one."""
    # Force PyTorch to only see the assigned physical GPU for this specific process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"-> Starting Seed {config['run']} on GPU {gpu_id}")
    run_experiment(config)
    print(f"<- Finished Seed {config['run']} on GPU {gpu_id}")


def run_multi_gpu(config_path, num_runs=4, num_gpus=4, stack_sizes=None):
    """Spawns concurrent processes to run experiments across multiple GPUs."""
    
    # CRITICAL: PyTorch requires 'spawn' for multiprocessing with CUDA.
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    with open(config_path, "r") as f:
        base_config = json.load(f)

    # --- THE FIX: Parse CUDA_VISIBLE_DEVICES ---
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_env:
        # User passed something like CUDA_VISIBLE_DEVICES=2,3
        # Parse it into a list: ['2', '3']
        gpu_list = [g.strip() for g in cuda_env.split(",") if g.strip()]
        print(f"Respecting CUDA_VISIBLE_DEVICES. Distributing across physical GPUs: {gpu_list}")
    else:
        # Fallback to default 0 to N-1
        gpu_list = [str(i) for i in range(num_gpus)]
        print(f"CUDA_VISIBLE_DEVICES not set. Defaulting to physical GPUs: {gpu_list}")

    num_available_gpus = len(gpu_list)
    processes = []
    
    for run in range(num_runs):
        # Round-robin assign runs strictly to the available GPUs in the list
        target_gpu_id = gpu_list[run % num_available_gpus] 
        
        # Deepish copy of the config for this specific run
        config = base_config.copy()
        config["run"] = run
        
        if stack_sizes is not None and run < len(stack_sizes):
            config["stack_size"] = stack_sizes[run]

        # Create and start the process
        p = mp.Process(target=worker_run, args=(target_gpu_id, config))
        p.start()
        processes.append(p)

    # Wait for all processes to finish before exiting the main script
    for p in processes:
        p.join()
        
    print("\nAll multi-GPU experiments completed successfully.")


def main(run=0, stack_size=None, config_path=None):
    """Single-run function for standard execution."""
    with open(config_path, "r") as f:
        config = json.load(f)

    config["run"] = run
    if stack_size is not None:
        config['stack_size'] = stack_size

    run_experiment(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to a JSON config file (without .json extension) in experiment_configs/.",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Flag to run 4 seeds concurrently across available GPUs."
    )
    args = parser.parse_args()

    config_dir = "experiment_configs"
    config_path = f"{config_dir}/{args.config}.json"

    if args.multi:
        print(f"Launching Multi-GPU execution for {args.config}...")
        run_multi_gpu(config_path, num_runs=4, num_gpus=4)
    else:
        print(f"Launching Single-GPU execution for {args.config}...")
        main(run=0, config_path=config_path)
