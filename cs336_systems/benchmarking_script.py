import logging
import os
from argparse import ArgumentParser
from contextlib import nullcontext
from datetime import datetime
from timeit import default_timer as timer
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd.profiler import record_function
from torch.cuda import nvtx
from torch.profiler import profile, ProfilerActivity

import cs336_basics
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

from cs336_basics.model import scaled_dot_product_attention
cs336_basics.model.scaled_dot_product_attention = scaled_dot_product_attention

logger = logging.Logger("benchmark")

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def initialize_model(
        d_model: int,
        d_ff: int,
        num_heads: int,
        num_layers: int,
        vocab_size: int=10_000,
        context_length: int=128,
        theta:float=10_000,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None
) -> BasicsTransformerLM:
    model = BasicsTransformerLM(
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        num_layers=num_layers,
        vocab_size=vocab_size,
        context_length=context_length,
        rope_theta=theta,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device is not None:
        model.to(device)
    return model


def get_random_batch(
        batch_size: int,
        vocab_size: int,
        context_length: int,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return inputs, targets


def nsys_profile(model, batch_data, warmup_steps, profile_steps,
                  pass_cond: Literal['forward','backward','optimizer'],
                  mixed_precision: bool = False):
    optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    if mixed_precision:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
        torch.set_float32_matmul_precision('high')
    inputs, targets = batch_data[0], batch_data[1]
    # inputs, targets = inputs.to(dtype), targets.to(dtype)
    for _ in range(warmup_steps):
        with nvtx.range("warmup step"):

            logits = model(inputs)
            if pass_cond == 'backward':
                loss = cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            torch.cuda.synchronize()
    # 测量阶段
    forward_times = []
    backward_times = []
    optimizer_times = []
    for i in range(profile_steps):
        with nvtx.range("profile step"):
            start_time = timer()
            output = model(inputs)
            torch.cuda.synchronize()
            end_time = timer()
            forward_times.append(end_time - start_time)
        if pass_cond == 'backward':
            with nvtx.range("backward step"):
                start_time = timer()
                loss = cross_entropy(output, targets)
                loss.backward()
                torch.cuda.synchronize()
                end_time = timer()
                backward_times.append(end_time - start_time)
        if pass_cond == 'optimizer':
            with nvtx.range("optimizer step"):
                start_time = timer()
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.synchronize()
                end_time = timer()
                optimizer_times.append(end_time - start_time)

        if pass_cond == 'forward':
            forward_mean = np.mean(forward_times)
            forward_std = np.std(forward_times)
            logger.info(f"Forward pass time: {forward_mean:.6f} +/- {forward_std:.6f}")
            return forward_mean, forward_std
        if pass_cond == 'backward':
            backward_mean = np.mean(backward_times)
            backward_std = np.std(backward_times)
            logger.info(f"Backward pass time: {backward_mean:.6f} +/- {backward_std:.6f}")
            return backward_mean, backward_std
        if pass_cond == 'optimizer':
            optimizer_mean = np.mean(optimizer_times)
            optimizer_std = np.std(optimizer_times)
            logger.info(f"Optimizer step time: {optimizer_mean:.6f} +/- {optimizer_std:.6f}")
            return optimizer_mean, optimizer_std

    return None, None, None


def profile_model(model,batch_data, warmup_steps, profile_steps,
                  pass_cond: Literal['forward','backward','all'],
                  mixed_precision: bool = False):
    mp = torch.autocast(device_type='cuda', dtype=torch.float16) if mixed_precision else nullcontext()
    with mp:
        for _ in range(warmup_steps):
            inputs, targets = batch_data[0], batch_data[1]
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            if pass_cond == 'backward' or pass_cond == 'forward':
                loss.backward()
            torch.cuda.synchronize()
        measurement_results = np.zeros(profile_steps)
        for i in range(profile_steps):
            if pass_cond == 'forward':
                start = timer()
                logits = model(batch_data[0])
                loss = cross_entropy(logits, batch_data[1])
            elif pass_cond == 'backward':
                logits = model(batch_data[0])
                loss = cross_entropy(logits, batch_data[1])
                torch.cuda.synchronize()
                start = timer()
                loss.backward()
            elif pass_cond == 'all':
                start = timer() # include the time for forward backward
                logits = model(batch_data[0])
                loss = cross_entropy(logits, batch_data[1])
                loss.backward()
            # else:
            #     raise ValueError(f'pass_cond {pass_cond} not recognized')
            torch.cuda.synchronize()
            end = timer()
            measurement_results[i] = end - start
        mean_time = np.mean(measurement_results)
        std = np.std(measurement_results)
    return mean_time, std


configs = {
    'small': {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12},
    'medium': {'d_model': 1024, 'd_ff': 4096, 'num_layers': 24, 'num_heads': 16},
    'large': {'d_model': 1280, 'd_ff': 5120, 'num_layers': 36, 'num_heads': 20},
    'xl': {'d_model': 1600, 'd_ff': 6400, 'num_layers': 48, 'num_heads': 25},
    '2.7B': {'d_model': 2560, 'd_ff': 10240, 'num_layers': 32, 'num_heads': 32},
}


def perform_all_profiles(include_warmup: bool, mixed_precision=False):
    inputs, targets = get_random_batch(args.batch_size, context_length=128, vocab_size=10_000)
    table_data : dict[str, list[float | str]] = {}
    for key, values in configs.items():
        model = initialize_model(**values)
        branch_cond : list[Literal['forward', 'backward', 'all']] = ['forward', 'backward', 'all']
        for pass_cond in branch_cond:
            mean_time, std = profile_model(model, (inputs, targets), warmup_steps=1 if include_warmup else 0,
                                           profile_steps=5, pass_cond=pass_cond, mixed_precision=mixed_precision)
            print(f"{key} {pass_cond} mean_time= {mean_time:.3f} seconds std_time= {std:.3f} seconds")
            analysis_time(key, mean_time, pass_cond, std, table_data)



def analysis_time(key, mean_time, pass_cond, std, table_data):
    if f'model_size' not in table_data:
        table_data[f'model_size'] = []
    if f'phase' not in table_data:
        table_data[f'phase'] = []
    if f'mean_time' not in table_data:
        table_data[f'mean_time'] = []
    if f'std' not in table_data:
        table_data[f'std'] = []
    table_data[f'model_size'].append(key)  # ✅ key 是 str，合法
    table_data[f'phase'].append(pass_cond)
    table_data[f'mean_time'].append(float(mean_time))  # ✅ 显式转为 float
    table_data[f'std'].append(float(std))
    df = pd.DataFrame(table_data)
    markdown_table = df.to_markdown(index=False)
    print(markdown_table)


def perform_nsys_profile(include_warmup, mixed_precision: bool = False):
    inputs, targets = get_random_batch(args.batch_size, context_length=128, vocab_size=10_000)
    tabel_data = {}
    for key, values in configs.items():
        model = initialize_model(**values)
        branch_cond : list[Literal['forward', 'backward', 'optimizer']] = ['forward', 'backward', 'optimizer']
        for pass_cond in branch_cond:
            mean_time, std = nsys_profile(model, (inputs, targets), warmup_steps=1 if include_warmup else 0,
                                           profile_steps=5, pass_cond=pass_cond, mixed_precision=mixed_precision)
            print(f"{key} {pass_cond} mean_time={mean_time:.3f} Seconds std_time={std:.3f} Seconds")
            analysis_time(key, mean_time, pass_cond, std, tabel_data)


def profile_memory(model, batch_data, context_length, pass_cond: str, mixed_precision: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if mixed_precision:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
        torch.set_float32_matmul_precision('high')
    model = model.to(dtype)
    optimizer = AdamW(model.parameters(),
          lr=1e-3,
          betas=(0.9, 0.95),
          eps=1e-8,
          weight_decay=0.01)
    inputs, targets = batch_data[0], batch_data[1]
    casting = torch.autocast(device_type='cuda', dtype=dtype) if mixed_precision else nullcontext()
    with casting:
        logger.info(f"================Starting memory for {pass_cond} pass  with context_length={context_length}=================")
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
        start_time = datetime.now()
        logger.info(f"{pass_cond} start time {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        # logger.info(f"Starting {pass_cond} {context_length} pass memory benchmarking")
        if pass_cond == 'forward':

            torch.cuda.reset_peak_memory_stats(device=device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logits = model(inputs)
            torch.cuda.synchronize()

        if pass_cond == 'all':
            # start_time = datetime.now()
            # logger.info(f"All pass start time {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            optimizer.zero_grad()
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            # end_time = datetime.now()
            # logger.info(f"All pass end time {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            # logger.info(f"All pass elapsed time {(end_time - start_time).total_seconds()} seconds")
            # logger.info(f"All pass peak memory {torch.cuda.max_memory_allocated(device) / (1024 ** 2)} MB")
        end_time = datetime.now()
        logger.info(f"{pass_cond} end time {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{pass_cond} elapsed time {(end_time - start_time).total_seconds()} seconds")

    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert bytes to MB
    logger.info(f"{pass_cond} peak memory = {peak_memory:.3f} MB")
    # memory_dir = "memory_files"
    # filename = f"memory_ctx{context_length}_{'forward' if pass_cond == 'forward' else 'backward'}_{'mixed' if mixed_precision else ''}.pickle"
    # full_path = os.path.join(memory_dir, filename)
    #
    # torch.cuda.memory._dump_snapshot(full_path)
    # logger.info(f"File saved to {full_path}")
    # torch.cuda.memory._record_memory_history(enabled=None)

    return peak_memory

def profile_memory_full_step(mixed_precision: bool=True):
    global results_data
    results = []
    for key, values in configs.items():
        for context_length in [128]: # 128, 256, 512, 1024
            model = initialize_model(**values, context_length=context_length,theta=10000)
            inputs, targets = get_random_batch(args.batch_size, context_length=context_length, vocab_size=10_000)
            branch_cond = ['forward', 'all']
            for pass_cond in branch_cond:
                logger.info(f"===== Memory Profiling model_size={key} context_length={context_length} {pass_cond} =====")
                results_data = profile_memory(model, (inputs, targets), context_length, pass_cond, mixed_precision=mixed_precision)
                logger.info(f"*****Results for model_size={key} context_length={context_length} {pass_cond} Peak_Memory={results_data:.3f}MB *****")
                row = {'model': key, 'context_length': context_length, 'pass_cond': pass_cond, 'peak_memory_MB': results_data}
                results.append(row)
            # Aggregate all results into a DataFrame and print a single LaTeX table
    df = pd.DataFrame(results)
    print("\nAggregated LaTeX Table:")
    print(df.to_latex(index=False, float_format="%.2f"))


class ToyModel(nn.Module):
    def __init__(self, in_features:int, out_features:int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=True)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(in_features, 10, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        logger.info(f"{self.fc1.weight.dtype=}")
        logger.info(f"{x.dtype}")
        x = self.fc1(x)
        logger.info(f"post fc1 x.dtyte={x.dtype}")
        x = self.relu(x)
        logger.info(f"post relu x.dtyte={x.dtype}")
        x = self.ln(x)
        logger.info(f"post ln x.dtyte={x.dtype}")
        x = self.fc2(x)
        logger.info(f"post fc2 x.dtyte={x.dtype}")
        return x

def test_mixed_precision():

    model = ToyModel(10, 10).to(torch.device('cuda'))
    x = torch.randn(1, 10).to(torch.device('cuda'))
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        logger.info(f"x.dtype={x.dtype}")
        y = model(x)
        logger.info(f"y.dtype={y.dtype}")

def run_benchmark_suite():
    timenow = datetime.now().strftime("%Y%m%d_%H%M%S")
    nsys_output_dir = f"nsys_output"
    os.makedirs(nsys_output_dir, exist_ok=True)
    print(f"[INFO] Recommended nsys command")
    print(f"nsys profile --output {nsys_output_dir}/profile_{timenow} python -m cs336_system.benchmarking_script")
    perform_all_profiles(include_warmup=True, mixed_precision=False)


def run_step(model, param, optimizer, enable_backward, enable_optimizer, mixed_precision):
    inputs, targets = param
    mps = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if mixed_precision else nullcontext()
    with mps:
        with record_function("forward"):
            logits = model(inputs)
            loss = cross_entropy(logits, targets)

        if enable_backward:
            with record_function("backward"):
                loss.backward()
            if enable_optimizer:
                with record_function("optimizer_step"):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        # with record_function("loss"):
        #     loss = cross_entropy(logits, targets)
        # if not enable_backward:
        #     return
        # with record_function("backward"):
        #     loss.backward()
        # if not enable_optimizer:
        #     return
        # with record_function("optimizer"):
        #     optimizer.step()
        #     optimizer.zero_grad(set_to_none=True)


def memory_profile(enable_backward: bool=True, enable_optimizer:bool=True, mixed_precision: bool=True):
    with (nullcontext() if enable_backward else torch.no_grad()):
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        n_steps = 3

        model = initialize_model(**configs['large'])
        device = model.lm_head.weight.device
        inputs, targets = get_random_batch(args.batch_size, context_length=128, vocab_size=10_000)
        optimizer = AdamW(model.parameters(),
                          lr=1e-3,
                          betas=(0.9, 0.95),
                          eps=1e-8,
                          weight_decay=0.01)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        if enable_backward:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        with profile(
            activities=[ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=n_steps),
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
            record_shapes=True,
            profile_memory = True,
            with_stack=True,
        ) as profiler:
            for _ in range(n_steps):
                run_step(model, (inputs, targets), optimizer, enable_backward, enable_optimizer, mixed_precision)
                profiler.step()
            # profiler.export_memory_timeline(f"timeline_{enable_backward}_{enable_optimizer}.html", device=device)
            profiler.export_chrome_trace(f"trace_{enable_backward}_{enable_optimizer}.json")
        torch.cuda.memory._dump_snapshot(f"memory_snapshot_{enable_backward}_{enable_optimizer}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--passes', type=str, default='both', choices=['forward', 'backward', 'both'])
    args = parser.parse_args()
    # perform_all_profiles(include_warmup=True, mixed_precision=False)
    # perform_nsys_profile(include_warmup=True, mixed_precision=False)
    # run_benchmark_suite()
    # profile_memory_full_step()
    memory_profile(enable_backward=True, enable_optimizer=True, mixed_precision=False)










