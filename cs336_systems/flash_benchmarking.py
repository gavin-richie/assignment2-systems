import pandas as pd
import torch
import triton.testing

from flash_attention import FlashAttentionTorch as FlashAttnPyTorch
from flash_attention import FlashAttentionTriton as FlashAttnTriton
from flash_attention import backward_pass_recomp as flash_attn_bkwd
from flash_attention import compiled_backward as flash_attn_bkwd_compiled
from utils import logger

def benchmark_flash_attn(context_length, d, dtype):
    batch_size = 1
    is_causal = False
    dtype=torch.float32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Q = torch.rand(batch_size, context_length, d, device=device, dtype=dtype)
    K = torch.rand(batch_size, context_length, d, device=device, dtype=dtype)
    V = torch.rand(batch_size, context_length, d, device=device, dtype=dtype)
    O = torch.rand(batch_size, context_length, d, device=device, dtype=dtype)

    L = torch.rand(batch_size, context_length, device=device, dtype=dtype)
    dO = torch.rand(batch_size, context_length, d, device=device, dtype=dtype)

    def forward_pytorch():
        FlashAttnPyTorch.apply(Q, K, V, is_causal)
    def forward_triton():
        FlashAttnTriton.apply(Q, K, V, is_causal)
    def backward_pytorch():
        flash_attn_bkwd(Q,K, V, O, L, dO, is_causal)
    def backward_triton():
        flash_attn_bkwd_compiled(Q,K, V, O, L, dO, is_causal)
    def fb_pytorch():
        output = FlashAttnPyTorch.apply(Q, K, V, is_causal)
        loss = torch.mean(output)
        loss.backward()
    def fb_triton():
        output = FlashAttnTriton.apply(Q, K, V, is_causal)
        loss = torch.mean(output)
        loss.backward()


    time_forward_pytorch = triton.testing.do_bench(forward_pytorch)
    logger.info(f"Forward pass time for PyTorch: {time_forward_pytorch: .2f} ms")
    time_forward_triton = triton.testing.do_bench(forward_triton)
    logger.info(f"Forward pass time for Triton: {time_forward_triton: .2f} ms")
    time_backward_pytorch = triton.testing.do_bench(backward_pytorch)
    logger.info(f"Backward pass time for PyTorch: {time_backward_pytorch: .2f} ms")
    time_backward_triton = triton.testing.do_bench(backward_triton)
    logger.info(f"Backward pass time for Triton: {time_backward_triton: .2f} ms")

    Q.requires_grad = True
    K.requires_grad = True
    V.requires_grad = True

    time_fb_pytorch = triton.testing.do_bench(fb_pytorch)
    logger.info(f"Forward-Backward pass time for PyTorch: {time_fb_pytorch: .2f} ms")
    time_fb_triton = triton.testing.do_bench(fb_triton)
    logger.info(f"Forward-Backward pass time for Triton {time_fb_triton: .2f} ms")

    return time_forward_triton, time_forward_pytorch, time_backward_triton, time_backward_pytorch, time_fb_triton, time_fb_pytorch


def create_flash_benchmark_table(benchmark_results, units="ms"):
    """Create and display a formatted table of Flash Attention benchmark results."""
    # Prepare data for DataFrame with multi-level index

    # First, collect all unique values for each dimension
    all_context_lengths = set()
    all_d_models = set()
    all_dtypes = set()

    for context_length in benchmark_results:
        all_context_lengths.add(context_length)
        for d in benchmark_results[context_length]:
            all_d_models.add(d)
            for dtype in benchmark_results[context_length][d]:
                all_dtypes.add(dtype)

    # Sort the values for consistent ordering
    all_d_models = sorted(all_d_models)
    all_dtypes = sorted(all_dtypes)
    all_context_lengths = sorted(all_context_lengths)

    rows = []
    # Change iteration order to: model_dim -> data_type -> context_length
    for d in all_d_models:
        for dtype in all_dtypes:
            for context_length in all_context_lengths:
                results = benchmark_results[context_length][d][dtype]
                row = {
                    'context_length': context_length,
                    'd_model': d,
                    'dtype': dtype,
                    'forward_triton': results['forward_triton'],
                    'forward_pytorch': results['forward_pytorch'],
                    'backward_triton': results['backward_triton'],
                    'backward_pytorch': results['backward_pytorch'],
                    'forward_backward_triton': results['fb_triton'],
                    'forward_backward_pytorch': results['fb_pytorch'],
                }
                rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)
    # Change index order to reflect new hierarchy: model_dim -> data_type -> context_length
    # df = df.set_index(['d_model', 'dtype', 'context_length'])

    # Format columns with units
    metric_columns = {
        'forward_triton': f'Forward Triton ({units})',
        'forward_pytorch': f'Forward PyTorch ({units})',
        'backward_triton': f'Backward Triton ({units})',
        'backward_pytorch': f'Backward PyTorch ({units})',
        'forward_backward_triton': f'F+B Triton ({units})',
        'forward_backward_pytorch': f'F+B PyTorch ({units})'
    }

    # Rename columns
    df = df.rename(columns=metric_columns)

    # Format values with 3 decimal places
    for col in metric_columns.values():
        df[col] = df[col].apply(lambda x: f"{x:.3f}")

    # Rename index names to reflect new hierarchy
    # df.index.names = ['Dimension', 'DType', 'Context Length']

    # Set pandas display options
    pd.set_option('display.max_colwidth', 20)
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', '{:.3f}'.format)

    logger.info(f"\n---------- Flash Attention Benchmark Results ----------")
    print(df.to_string( index = False,
                        col_space = 12,  # 可适当调小，如果 index 较宽
                        justify = 'center',  # 让所有列左对齐（默认一般是左对齐，但明确指定更可控）
                        sparsify = False,  # 避免 MultiIndex 中相同值被省略（可选）
    ))

    markdown_table = df.to_markdown(index=False)
    print(markdown_table)
    return df

if __name__ == '__main__':
    context_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]  # 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
    ds = [16, 32, 64]  # 32, 64, 128
    dtypes = ["float32", "bfloat16"]

    benchmark_results = {}

    for context_length in context_lengths:
        benchmark_results[context_length] = {}
        for d in ds:
            benchmark_results[context_length][d] = {}
            for dtype in dtypes:
                logger.info(f"Benchmark Settings: context_length={context_length}, d={d},dtype={dtype}.")

                ms_forward_triton, ms_forward_pytorch, ms_backward_triton, ms_backward_pytorch, ms_fb_triton, ms_fb_pytorch = benchmark_flash_attn(context_length, d, dtype)
                torch._dynamo.reset()

                benchmark_results[context_length][d][dtype] = {
                    'forward_triton': ms_forward_triton,
                    'forward_pytorch': ms_forward_pytorch,
                    'backward_triton': ms_backward_triton,
                    'backward_pytorch': ms_backward_pytorch,
                    'fb_triton': ms_fb_triton,
                    'fb_pytorch': ms_fb_pytorch,
                    }

    create_flash_benchmark_table(benchmark_results, "ms")























