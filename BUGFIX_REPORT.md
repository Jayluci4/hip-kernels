# Gibberish Output Bug: hipBLASLt Stream Ordering Race Condition

## Summary

The HIP port of the GPT-OSS inference engine on AMD MI300X (ROCm 7.1.0) produced gibberish text for both the 20B and 120B models. The root cause was a **stream synchronization race condition** between custom MXFP4 dequantization HIP kernels and hipBLASLt GEMM calls on the same compute stream. Despite being submitted to the same HIP stream — which should guarantee serial execution — hipBLASLt matmul operations were reading from dequantization staging buffers before the preceding dequant kernel had finished writing to them. The fix was adding explicit `hipStreamSynchronize()` barriers between dequant kernels and hipBLASLt GEMM calls.

## The Engine Architecture

The GPT-OSS model uses a Mixture-of-Experts (MoE) architecture where each transformer layer contains:

1. **RMSNorm** → **Self-Attention** → residual add
2. **RMSNorm** → **Router** → **Top-K Expert Selection** → **Expert FFN** → weighted combine + residual add

Each expert FFN consists of:
- **W1 (gate_up_proj)**: MXFP4-quantized weight matrix `[5760, 2880]` — dequantized on-the-fly
- **MLP1 GEMM**: `tokens × W1^T` producing gate+up activations
- **Bias + SwiGLU activation**
- **W2 (down_proj)**: MXFP4-quantized weight matrix `[2880, 2880]` — dequantized on-the-fly
- **MLP2 GEMM**: `swiglu_out × W2^T` producing expert output
- **Bias**

Experts are processed in **batches of 8** (`GROUPED_BATCH_SIZE=8`) to amortize kernel launch overhead. For the 20B model (32 experts), this means 4 batches per layer. For 120B (128 experts), 16 batches.

The batch processing loop in `src/model/moe_layer.cu` (starting ~line 885) follows this pattern for each batch:

```
1. Populate MxFp4BatchDesc descriptors for W1
2. hipMemcpy descriptors to device (synchronous)
3. Launch mxfp4_dequant_batched kernel → writes to stg.dequant[0][s]
4. Build GEMM arrays: mlp1_B[s] = stg.dequant[0][s]
5. Call hipblasLtMatmul via gemm_bf16_lt_multi → reads stg.dequant[0][s]
6. Bias + SwiGLU
7. Repeat steps 1-5 for W2
8. Bias
```

## Symptoms

### 120B Model (Original Discovery)

The 120B model appeared to produce **zero MoE deltas** — `after_moe == after_attn` for all 36 layers. This was initially diagnosed as "expert FFN outputs not being added to the residual stream" and classified as a model logic bug since it reproduced on both CUDA (H100) and HIP (MI300X).

This diagnosis was **incorrect**. The MoE deltas were not zero — they were filled with garbage values from stale dequant buffers, which happened to look like near-zero deltas when the race condition produced certain patterns of corrupted weights.

### 20B Model (Detailed Investigation)

When the 20B model was brought up, it produced completely incoherent gibberish:

```
Generated: 'ied(te/peraziai stairprox ret494iskol-comremosritter form Kernel,APH-Mar162...'
Generated: '-fer:\n blobganiPremensثق ру rv justlay PS flagszzle473ample emergencyPN...'
```

The RMS values exploded across layers: L0=2.86, L1=4.06, L2=7.94, L3=34.83, L4=59.97, ... L23=13111.

## Investigation Process

### Step 1: Isolate Divergence Point (`hf_hooks_debug.py`)

We used PyTorch hooks on the HuggingFace reference model to capture intermediate activations at every sub-layer boundary. Comparing HF reference values with engine diagnostics:

| Checkpoint | HF Reference | Engine | Match? |
|---|---|---|---|
| Embedding | ✓ | ✓ | ✓ |
| L0 after_attn | `[0.268, -0.264, -0.879, ...]` | `[0.268, -0.266, -0.879, ...]` | ✓ |
| L0 normed2 | `[0.180, -0.135, -0.414, ...]` | `[0.180, -0.136, -0.414, ...]` | ✓ |
| L0 router top-4 | experts `[5,9,6,18]` | experts `[5,9,6,18]` | ✓ |
| L0 router weights | `[0.4720, 0.3169, 0.1078, 0.1033]` | `[0.4720, 0.3169, 0.1078, 0.1033]` | ✓ |
| **L0 moe_delta** | `[0.159, 0.122, -0.049, ...]` | `[-0.008, 0.108, -0.117, ...]` | **✗ DIVERGES** |

Everything was correct up to and including the router — the divergence was specifically in the expert FFN computation.

### Step 2: Manual Expert Computation (`hf_expert_debug3.py`)

We wrote a Python script that manually dequantized MXFP4 weights from the raw safetensor files and performed FP32 matrix multiplications to compute each expert's output independently.

The MXFP4 dequantization logic:
- Each weight block contains 16 packed bytes (32 E2M1 nibble pairs) plus 1 E8M0 scale byte
- E2M1 LUT: `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]`
- E8M0 scale: `2^(scale_byte - 127)`
- Dequantized value: `LUT[nibble] × scale`

The manual Python computation matched HF perfectly:
```
Manual combined first8: [0.160, 0.121, -0.049, 0.124, 0.984, -0.187, -0.243, 0.062]
HF moe_delta first8:    [0.159, 0.122, -0.049, 0.125, 0.984, -0.188, -0.244, 0.062]
```

### Step 3: Per-Expert Comparison — The Batch Boundary Pattern

Comparing individual expert outputs between the manual Python computation and the engine revealed the critical pattern:

| Expert | Batch | Manual (Python) | Engine | Match? |
|---|---|---|---|---|
| Expert 5 | Batch 0 (experts 0-7) | `[0.125, 0.357, -0.007, 0.128]` | `[0.125, 0.354, -0.006, 0.129]` | ✓ |
| Expert 6 | Batch 0 (experts 0-7) | `[0.221, 0.238, 0.299, -0.104]` | `[0.221, 0.235, 0.299, -0.104]` | ✓ |
| **Expert 9** | **Batch 1 (experts 8-15)** | `[0.181, -0.194, -0.196, 0.220]` | `[-0.209, -0.110, -0.041, 0.106]` | **✗ WRONG** |
| **Expert 18** | **Batch 2 (experts 16-23)** | `[0.189, -0.112, -0.157, 0.050]` | `[-0.231, -0.486, -1.297, 0.477]` | **✗ WRONG** |

**Batch 0 experts were correct. Batch 1+ experts were completely wrong.** This pointed directly at the batched processing loop.

### Step 4: Dequant Buffer Verification

We added diagnostic code to `moe_layer.cu` to dump the dequantized W1 values for expert 9 at two different points:

1. **Immediately after W1 dequant** (before GEMM):
   ```
   [E9 W1-FRESH] dequant[0][1] first8=[-0.031250,-0.031250,-0.000000,-0.046875,0.015625,0.015625,-0.000000,-0.031250]
   ```
   This **matched** the manual Python dequant: `[-0.03125, -0.03125, -0.0, -0.046875, 0.015625, 0.015625, -0.0, -0.03125]` ✓

2. **After the full batch** (after W2 dequant + MLP2 GEMM):
   ```
   [E9 DIAG] W1_dequant first8=[-0.003906,-0.003906,0.000000,0.000000,-0.003906,0.003906,0.007812,-0.007812]
   ```
   This was the W2 dequant data (the buffer had been reused), confirming the dequant buffer was overwritten as expected.

The raw packed bytes on the device matched the safetensor source data exactly:
```
Engine:      raw packed[0..3]=[aa,b8,11,a8] scale=122
Safetensor:  raw packed[0..3]=['0xaa', '0xb8', '0x11', '0xa8'] scale=122
```

**The weights were loaded correctly. The dequant kernel produced correct output. But the GEMM was reading stale data.**

### Step 5: The Stream Synchronization Fix

Adding `hipStreamSynchronize(compute_stream_)` between the dequant kernel and the GEMM call fixed expert 9 (batch 1). But expert 18 (batch 2) was still wrong — because the sync was only added for batch 1's diagnostic path.

Adding the sync before **every** GEMM call (both MLP1 and MLP2) across all batches fixed everything:

```
Expert 5  (batch 0): [0.1250, 0.3535, -0.0063, 0.1289] ✓
Expert 9  (batch 1): [0.1836, -0.1973, -0.1982, 0.2207] ✓ (was [-0.5742, 0.0222, 1.1250, -0.0791])
Expert 6  (batch 0): [0.2207, 0.2354, 0.2988, -0.1040] ✓
Expert 18 (batch 2): [0.1875, -0.1113, -0.1562, 0.0498] ✓ (was [-0.2305, -0.4863, -1.2969, 0.4766])
```

Generation output became correct:
```
Generated: 'Paris.'
Generated: '4'
Generated: 'Hello! I'm doing great—thanks for asking. How about you?'
```

## Root Cause

**hipBLASLt on ROCm 7.1.0 does not properly respect HIP stream ordering after custom HIP kernels on MI300X (gfx942).**

The sequence on `compute_stream_` is:
1. Custom HIP kernel (`mxfp4_dequant_kernel_batched`) writes to `stg.dequant[0][s]`
2. `hipblasLtMatmul()` reads from `stg.dequant[0][s]` as its B matrix

Both operations are submitted to the same HIP stream. Per the HIP/CUDA programming model, operations on the same stream execute in submission order — operation 2 should not begin until operation 1 completes. However, on MI300X with ROCm 7.1.0, the hipBLASLt matmul appears to begin execution before the preceding custom kernel's global memory writes are visible, causing it to read stale or partially-written data from the dequant staging buffers.

This manifests as:
- **Batch 0 experts often appearing correct**: The first batch's dequant has no prior GEMM on the stream, and the synchronous `hipMemcpy` for batch descriptors at the start of the loop acts as an implicit barrier
- **Batch 1+ experts producing garbage**: The GEMM reads from dequant buffers that haven't been fully written yet

The issue may be related to L2 cache coherence across the 8 GCDs (Graphics Compute Dies) on MI300X, where writes from one GCD's CUs may not be visible to another GCD's CUs without an explicit memory fence, even on the same stream.

## The Fix

**File: `src/model/moe_layer.cu`**

Two `hipStreamSynchronize(compute_stream_)` calls were added — one after each dequant kernel launch, before the corresponding GEMM:

### After W1 (gate_up_proj) dequant (~line 912):
```cpp
mxfp4_dequant_batched(d_dequant_descs_, bi.active_count,
                       mlp1_elements, compute_stream_);
PROF_END(device_id_, compute_stream_);

// ROCm workaround: hipBLASLt may not properly respect stream
// ordering after custom HIP kernels on MI300X.  Without this
// barrier the GEMM can read stale dequant buffers.
CUDA_CHECK(hipStreamSynchronize(compute_stream_));

// -- Build MLP1 GEMM arrays --
```

### After W2 (down_proj) dequant (~line 977):
```cpp
mxfp4_dequant_batched(d_dequant_descs_, bi.active_count,
                       mlp2_elements, compute_stream_);
PROF_END(device_id_, compute_stream_);

// ROCm workaround: same as W1 above
CUDA_CHECK(hipStreamSynchronize(compute_stream_));

// -- Build MLP2 GEMM arrays (reads from bank[0]) --
```

### Performance Impact

Each `hipStreamSynchronize` blocks the CPU until the GPU completes all pending work on the stream. This adds:
- 2 syncs per batch × ~4 batches (20B, 32 experts) = 8 syncs per MoE layer × 24 layers = **192 syncs per forward pass** (20B)
- 2 syncs per batch × ~16 batches (120B, 128 experts) = 32 syncs per MoE layer × 36 layers = **1152 syncs per forward pass** (120B)

This reduces GPU utilization by preventing the CPU from queuing work ahead of the GPU. A future optimization could investigate lighter-weight barriers (e.g., `hipEventRecord` + `hipStreamWaitEvent` on an internal synchronization stream) or filing a bug with AMD to fix the underlying hipBLASLt stream ordering issue.

## Verification

### 20B Model
```
Prompt: "The capital of France is" → "Paris."
Prompt: "What is 2 + 2?"          → "4"
Prompt: "Hello, how are you?"     → "Hello! I'm doing great—thanks for asking."
```

### 120B Model
```
Prompt: "The capital of France is" → "The capital of France is **Paris**."
Prompt: "What is 2 + 2?"          → "2 + 2 = 4."
Prompt: "Hello, how are you?"     → "Hello! I'm doing great, thanks for asking. How..."
```

Both models produce coherent, factually correct output with the fix applied.

## Files Referenced

| File | Purpose |
|---|---|
| `src/model/moe_layer.cu` | MoE layer implementation — contains the bug and fix |
| `src/kernels/mxfp4_dequant.cu` | MXFP4 dequantization kernels (batched + non-batched) |
| `include/cuda_utils.h` | `gemm_bf16_lt_multi()` — hipBLASLt multi-GEMM wrapper |
| `include/config.h` | Model architecture constants |
| `src/memory/weight_loader.cu` | MXFP4 weight loading and per-expert slicing |
| `hf_hooks_debug.py` | HF reference model hook-based intermediate capture |
| `hf_expert_debug.py` | Initial expert structure exploration |
| `hf_expert_debug2.py` | Manual MXFP4 dequant + single expert computation |
| `hf_expert_debug3.py` | Full 4-expert manual computation with per-expert comparison |
| `test_inference.py` | 120B model inference test |
| `test_inference_20b.py` | 20B model inference test |

## Timeline

1. **120B model**: Observed gibberish output, diagnosed as "zero MoE delta" — initially attributed to model logic bug
2. **20B model brought up**: Same gibberish, but with 32 experts (vs 128) making debugging tractable
3. **Hook-based comparison**: Isolated divergence to L0 MoE delta (attention + routing correct)
4. **Manual MXFP4 computation**: Confirmed Python dequant+GEMM matches HF exactly
5. **Per-expert comparison**: Discovered batch 0 correct, batch 1+ wrong
6. **Dequant buffer inspection**: Confirmed weights loaded correctly, dequant output correct
7. **Stream sync experiment**: `hipStreamSynchronize` between dequant and GEMM fixes all batches
8. **Fix applied to both W1 and W2 transitions**: Both 20B and 120B models produce correct output
