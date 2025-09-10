import os
import time
from torch.multiprocessing import Queue

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_systems.utils import logger


def setup(rank, world_size, device):
    if device.type == 'cuda':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend='nccl',
                                init_method="env://?use_libuv=False",
                                rank=rank,
                                world_size=world_size)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend='gloo',
                                init_method="env://?use_libuv=False",
                                rank=rank,
                                world_size=world_size)


def benchmark_allreduce(rank, world_size, warmup_steps, benchmark, device: str, data_size, result_queue):
    setup(rank, world_size, torch.device(device, rank) if device == 'cuda' else torch.device(device))
    device_obj = torch.device(device, rank) if device == 'cuda' else torch.device(device)
    logger.info(f"Rank {rank} is using device {device} device_obj {device_obj}")

    numel = data_size * 1024 * 1024 // 4
    data = torch.randn(numel,dtype=torch.float32, device=device_obj)

    for _ in range(warmup_steps):
        dist.all_reduce(data, async_op=False)
        if device_obj.type == 'cuda':
            torch.cuda.synchronize(device_obj)
    times = []

    for _ in range(benchmark):
        data.uniform_()
        if device == 'cuda':
            torch.cuda.synchronize(device_obj)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        start_time = time.time()
        logger.info(f"Rank {rank} data (before all_reduce): {data} {data.shape}")
        dist.all_reduce(data, async_op=False)
        logger.info(f"Rank {rank} data (after all_reduce): {data} {data.shape}")
        if device_obj == 'cuda':
            torch.cuda.synchronize(device_obj)
            end.record()
        end_time = time.time()

        if device == 'cuda':
            logger.info(f"All Reduce time: {start.elapsed_time(end)}")

        times.append((end_time - start_time)*1000) # convert to milliseconds

    mean_time = np.mean(times)
    std_time = np.std(times)

    record = {
        'backend': 'nccl' if device_obj.type == 'cuda' else 'gloo',
        'device_type': device_obj.type,
        'world_size': world_size,
        'data_size': data_size,
        'time_ms': mean_time,
        'time_std_ms': std_time,
    }

    if rank == 0:
        all_records = [None] * world_size
        all_records[0] = record
    else:
        all_records = None

    record_tensor = torch.tensor([
        float(mean_time),
        float(std_time),
    ], dtype=torch.float64, device=device_obj)

    if device_obj.type == 'cuda':
        gathered_tensors = [torch.zeros_like(record_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, record_tensor)
        if rank == 0:
            for i, tensor in enumerate(gathered_tensors[1:], 1):
                all_records[i] = {
                    'backend': 'nccl' if device_obj.type == 'cuda' else 'gloo',
                    'device_type': device_obj.type,
                    'world_size': world_size,
                    'data_size': data_size,
                    'time_ms': tensor[0].item(),
                    'time_std_ms': tensor[1].item(),
                }
    else:
        gathered_tensors = [torch.zeros_like(record_tensor) for _ in range(world_size)]
        dist.gather(record_tensor, gather_list=gathered_tensors if rank == 0 else None, dst=0)
        if rank == 0:
            for i, tensor in enumerate(gathered_tensors[1:], 1):  # Skip rank 0 as we already have it
                all_records[i] = {
                    'backend': 'nccl' if device_obj.type == 'cuda' else 'gloo',
                    'device_type': device_obj.type,
                    'world_size': world_size,
                    'data_size': data_size,
                    'time_ms': tensor[0].item(),
                    'time_std_ms': tensor[1].item()
                }

    dist.destroy_process_group()
    if rank == 0:
        result_queue.put(all_records)


if __name__ == '__main__':
    world_sizes = [2]
    warmup_steps = 5
    benchmark = 3
    data_sizes = [1, 10, 100, 1000] # 100
    device ='cpu'
    all_results = []

    for world_size in world_sizes:
        for data_size in data_sizes:
            result_queue = Queue()
            mp.spawn(benchmark_allreduce,
                     args=(world_size, warmup_steps, benchmark,device, data_size, result_queue),
                     nprocs=world_size,
                     join=True)
            try:
                results = result_queue.get(block=False, timeout=5)  # Add timeout to prevent hanging
                if results:
                    all_results.extend(results)  # Use extend() instead of append() to flatten the list
                else:
                    print(f"Worker failed")
            except Exception as e:
                logger.info(f"Failed to retrieve results from queue: {e}")

        # Create DataFrame from results and print summary
    if all_results:
        df = pd.DataFrame(all_results)
        # Group by relevant columns and calculate statistics
        summary = df.groupby(['backend', 'device_type', 'world_size', 'data_size']).agg({
            'time_ms': ['mean', 'std']
        }).reset_index()

        # Flatten column names
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

        # Format the output for better readability
        print("\nBenchmark Results Summary (times in milliseconds):")
        # print(summary.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
        # print("\nLaTeX Table:")
        # print(summary.to_latex(index=False, float_format=lambda x: f'{x:.2f}'))
        markdown_table = summary.to_markdown(index=False)
        print(markdown_table)
