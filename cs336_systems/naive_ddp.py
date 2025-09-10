import os
import sys
import time
from os.path import exists

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.ddp_config import TrainConfig, generate_random_data, set_seed, load_checkpoint, save_checkpoint, \
    gradient_clipping
from cs336_systems.utils import logger
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW


def train_regular(input_data, target_data, config, device):
    if config.seed is not None:
        set_seed(config.seed)

    model = BasicsTransformerLM(config.vocab_size,
                             config.context_length,
                             config.d_model,
                             config.num_layers,
                             config.num_heads,
                             config.d_ff,
                             config.rope_theta)

    model.to(device)

    if os.path.exists('initial_params.pt'):
        load_checkpoint(model)
    else:
        for param in model.parameters():
            param.data = torch.randn_like(param.data)
        save_checkpoint(model)

    optimizer = AdamW(model.parameters(),
                      lr=config.learning_rate,
                      betas=(config.beta1, config.beta2),
                      eps=config.epsilon,
                      weight_decay=config.weight_decay,)

    input_data = input_data.t().to(device)
    target_data = target_data.t().to(device)

    forward_time = []
    losses = []
    backward_time = []
    full_step_time = []
    send_grad_time = []

    for step in range(config.warmup_steps):
        logits = model(input_data)
        loss = cross_entropy(logits, target_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_start_time = time.time()
    # for step in range(config.num_steps):
    for step in tqdm(range(1, config.num_steps+1)):
        step_start_time = time.time()

        forward_start_time = time.time()
        logits = model(input_data)
        forward_end_time = time.time()

        loss = cross_entropy(logits, target_data)
        losses.append(loss.item())

        backward_start_time = time.time()
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), config.gradient_clipping)
        backward_end_time = time.time()

        optimizer.step()

        full_step_time.append(time.time() - step_start_time)
        forward_time.append(forward_end_time - forward_start_time)
        backward_time.append(backward_end_time - backward_start_time)
        send_grad_time.append(0.0)

        if step % 5 == 0:
            logger.info(f"Step {step}, Loss: {loss.item():.4f}")

    train_end_time = time.time()
    mean_times = {
        'forward_time': np.mean(forward_time)*1000, # convert to milliseconds
        'backward_time': np.mean(backward_time),
        'full_step_time': np.mean(full_step_time),
        'send_grad_time': np.mean(send_grad_time),
    }
    time_df = pd.DataFrame({
        'Metrics':['Forward', 'Backward', 'GradientCommunication', 'FullStep'],
        'Time(ms)':[mean_times['forward_time'],
                    mean_times['backward_time'],
                    mean_times['send_grad_time'],
                    mean_times['full_step_time']
                    ],
        'Processes': [1]*4,
        'Batch_size': [config.batch_size]*4,
    })

    time_df.to_csv('regular_time_results.csv', index=False)

    logger.info(f"Regular Training Time Results:")
    print(time_df.to_string(index=False))

    torch.save(model.state_dict(), 'regular_final_params.pt')
    torch.save({
        'losses': losses,
        'final_loss': losses[-1],
        'training_time': train_end_time-train_start_time,
        'timing_stats': {
            'forward_time_ms': mean_times['forward_time'],
            'backward_time_ms': mean_times['backward_time'],
            'grad_comm_time_ms': mean_times['send_grad_time'],
            'total_step_time_ms': mean_times['full_step_time']
        }
    }, 'regular_results.pt')

    return {
        'model': model,
        'losses': losses,
        'final_loss': losses[-1],
        'training_time': train_end_time-train_start_time,
        'params': {name: param.data.clone() for name, param in model.named_parameters()},
        'timing_stats': mean_times
    }


def setup(rank, world_size, device):
    if device.type == 'cuda':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        if sys.platform == 'win32':
            backend = 'gloo'
            init_method = 'env://?use_libuv=False'
        else:
            backend = 'nccl'
            init_method = 'env://'
        dist.init_process_group(backend,
                                init_method=init_method,
                                rank=rank, world_size=world_size)

    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        dist.init_process_group("gloo",
                                init_method="env://?use_libuv=False",
                                rank=rank,
                                world_size=world_size)


def train_naive_ddp(rank, world_size, input_data, target_data, model, config, device, flatten: bool=False):
    try:
        if config.seed is not None:
            set_seed(config.seed)
        setup(rank, world_size, device)
        logger.info(f"rank:{rank} setup done.")
        if device.type == 'cuda':
            model.cuda(rank)
        else:
            model.to(device)

        if rank == 0:
            if os.path.exists("ddp_initial_params.pt"):
                load_checkpoint(model)
            else:
                for param in model.parameters():
                    param.data = torch.randn_like(param.data)
                save_checkpoint(model)

        # parameter broadcast
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

        if rank == 0:
            logger.info(f"Broadcast parameters from rank {rank} to all processes.")
        dist.barrier()

        local_batch_size = config.batch_size // world_size
        local_start_index = rank * local_batch_size
        end_start_index = local_start_index + local_batch_size

        if device.type == 'cuda':
            local_input_data = input_data[:, local_start_index:end_start_index].cuda(rank)
            local_target_data = target_data[:, local_start_index:end_start_index].cuda(rank)
        else:
            local_input_data = input_data[:, local_start_index:end_start_index].to(device)
            local_target_data = target_data[:, local_start_index:end_start_index].to(device)

        # Transpose the input data to match the expected shape (batch_size, seq_len, d_model)
        local_input_data = local_input_data.transpose(0, 1)  # (batch_size, seq_len)
        local_target_data = local_target_data.transpose(0, 1)
        logger.info(f"local_input_data.shape:{local_input_data.shape}, local_target_data.shape:{local_target_data.shape}")

        optimizer = AdamW(model.parameters(),
                          lr=config.learning_rate,
                          betas=(config.beta1, config.beta2),
                          eps=config.epsilon,
                          weight_decay=config.weight_decay)

        forward_time = []
        backward_time = []
        full_step_time = []
        send_grad_time = []
        losses = []

        start_train_time = time.time()
        for step in range(config.warmup_steps):
            logits = model(local_input_data)
            loss = cross_entropy(logits, local_target_data)

            optimizer.zero_grad()
            loss.backward()
            gradient_clipping(model.parameters(), config.gradient_clipping)
            optimizer.step()

        for step in tqdm(range(1, config.num_steps+1), desc="Training Steps"):
            step_start_time = time.time()
            forward_start_time = time.time()
            logits = model(local_input_data)
            forward_time.append(time.time() - forward_start_time)

            loss = cross_entropy(logits, local_target_data)

            optimizer.zero_grad()
            backward_start_time = time.time()
            loss.backward()
            gradient_clipping(model.parameters(), config.gradient_clipping)
            backward_time.append(time.time() - backward_start_time)

            send_grad_start_time = time.time()
            if not flatten:

                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= world_size
                send_grad_time.append(time.time() - send_grad_start_time)
            else:
                param_list = model.state_dict()
                flat_grads = torch._utils._flatten_dense_tensors([param.grad.data for param in model.parameters()])
                dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=False)
                flat_grads /= world_size
                unflat_grads = torch._utils._unflatten_dense_tensors(flat_grads, [tensor for tensor in param_list.values()])

                for param, tensor in zip(model.parameters(), unflat_grads):
                    param.grad.data = tensor

            optimizer.step()
            full_step_time.append(time.time() - step_start_time)


            losses.append(loss.item())
            if step % config.log_interval == 0:
                logger.info(f"Step {step}: Loss = {loss.item():.4f}")
                logger.info(f"Forward Time = {forward_time[-1]:.4f}, Backward Time = {backward_time[-1]:.4f}, Full Step Time = {full_step_time[-1]:.4f}, Send Grad Time = {send_grad_time[-1]:.4f}")

        train_elapsed_time = time.time() - start_train_time
        logger.info(f"Elapsed Time = {train_elapsed_time:.4f}")

        mean_times = {
            'forward_time': np.mean(forward_time),
            'backward_time': np.mean(backward_time),
            'full_step_time': np.mean(full_step_time),
            'send_grad_time': np.mean(send_grad_time),
        }

        times_tensor = torch.tensor([
            mean_times['forward_time'],
            mean_times['backward_time'],
            mean_times['full_step_time'],
            mean_times['send_grad_time'],
        ], device=device)

        # create a tensor to store the mean times from all processes
        if rank == 0:
            gathered_times = [torch.zeros_like(times_tensor) for _ in range(world_size)]
        else:
            gathered_times = None

        dist.gather(times_tensor, gathered_times, dst=0)

        if rank == 0:
            # calculate the mean of the gathered times
            gathered_times = torch.stack(gathered_times)
            avg_times = gathered_times.mean(dim=0)
            time_df = pd.DataFrame({
                'Metrics':['Forward', 'Backward', 'FullStep', 'SendGrad'],
                'Time(ms)':avg_times.tolist(),
                'Processes': [world_size]*4,
                'Batch_size': [config.batch_size]*4,
            })

            # Save timing results to CSV
            time_df.to_csv('ddp_time_results.csv', index=False)

            # Print timing table
            print("\nTiming Results:")
            print(time_df.to_string(index=False))

            # Save final parameters and results
            torch.save(model.state_dict(), 'ddp_final_params.pt')
            torch.save({
                'losses': losses,
                'final_loss': losses[-1],
                'training_time': train_elapsed_time,
                'timing_stats': {
                    'forward_time_ms': avg_times[0].item(),
                    'backward_time_ms': avg_times[1].item(),
                    'grad_comm_time_ms': avg_times[2].item(),
                    'total_step_time_ms': avg_times[3].item()
                }
            }, 'ddp_results.pt')

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()



def naive_ddp_benchmarking():

    config = TrainConfig()

    os.makedirs(os.path.dirname('data'), exist_ok=True)

    if os.path.exists('ddp_random_data.pt'):
        input_data = torch.load('ddp_random_data.pt')
        target_data = torch.load('ddp_random_data.pt')
    else:
        input_data = generate_random_data(config.batch_size, config.context_length, config.vocab_size, config.seed)
        target_data = generate_random_data(config.batch_size, config.context_length, config.vocab_size, config.seed)

    if sys.platform == 'linux':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # 根据平台调整 world_size，避免在不支持的平台上创建过多进程
    if device.type == 'cpu':
        # 在 CPU 上运行时，限制进程数不超过可用 CPU 核心数的一半

        cpu_count = os.cpu_count()
        config.world_size = min(config.world_size, max(1, cpu_count // 2))
        logger.info(f"在 CPU 上运行，已将 world_size 调整为 {config.world_size}")

    # logger.info(f"Starting Regular Training...")
    # regular_results = train_regular(input_data, target_data, config, device)


    model = BasicsTransformerLM(config.vocab_size,
                             config.context_length,
                             config.d_model,
                             config.num_layers,
                             config.num_heads,
                             config.d_ff,
                             config.rope_theta)
    model.to(device)

    logger.info(f"Starting Naive DDP Training...")
    mp.spawn(train_naive_ddp, args=(config.world_size, input_data, target_data, model, config, device),
             nprocs=config.world_size,
             join=True)


def benchmark_compare(regular_results, ddp_results):
    """Print comparison of regular and DDP training results"""
    print("\n" + "=" * 50)
    print("Training Results Comparison")
    print("=" * 50)

    print("\nLoss Comparison:")
    print(f"Regular Training Final Loss: {regular_results['final_loss']:.4f}")
    print(f"DDP Training Final Loss: {ddp_results['final_loss']:.4f}")
    print(f"Loss Difference: {abs(regular_results['final_loss'] - ddp_results['final_loss']):.4f}")

    print("\nTraining Time:")
    print(f"Regular Training Time: {regular_results['training_time']:.2f} seconds")
    print(f"DDP Training Time: {ddp_results['training_time']:.2f} seconds")
    print(f"Speedup: {regular_results['training_time'] / ddp_results['training_time']:.2f}x")

    print("\nParameter Comparison:")
    max_diff = 0
    for name in regular_results['params'].keys():
        diff = torch.max(torch.abs(regular_results['params'][name] - ddp_results['params'][name]))
        max_diff = max(max_diff, diff.item())
    print(f"Maximum Parameter Difference: {max_diff:.6f}")
    print("=" * 50 + "\n")


def compare_results():
    config = TrainConfig()
    regular_results = torch.load('regular_results.pt')
    ddp_results = torch.load('ddp_results.pt')
    ddp_model = BasicsTransformerLM(config.vocab_size,
                             config.context_length,
                             config.d_model,
                             config.num_layers,
                             config.num_heads,
                             config.d_ff,
                             config.rope_theta)
    ddp_model.load_state_dict(torch.load('ddp_model_params.pt'))
    ddp_results['params'] = {name: param.data.clone() for name, param in ddp_model.named_parameters()}

    benchmark_compare(regular_results, ddp_results)


if __name__ == '__main__':
    naive_ddp_benchmarking()
