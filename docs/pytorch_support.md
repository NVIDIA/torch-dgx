# Supported Layers and Operators

## Supported Layers

Supported `torch.nn` modules include:

- `Linear`
- `Conv2d`
- `Embedding`
- `Dropout`
- `Unfold`
- `ReflectionPad2d`
- `ReLU`
- `GELU`
- `Sigmoid`
- `SiLU`
- `Softmax`
- `LogSoftmax`
- `Tanh`
- `Threshold`
- `GroupNorm`
- `LayerNorm`
- `L1Loss`
- `MSELoss`
- `NLLLoss`
- `CrossEntropyLoss` with class-index targets and `label_smoothing=0`
- `UpsamplingNearest2d`

## Scaled Dot-Product Attention

SDPA uses CUTLASS forward and backward kernels for dense FP32 Q/K/V tensors
shaped `[batch, heads, sequence, channels]`, including strided and broadcast
inputs. The fused path supports `is_causal=True` with top-left alignment for
rectangular attention, or a boolean/FP32 additive mask with two to four
dimensions broadcastable to `[batch, heads, queries, keys]`. Boolean `True`
entries participate in attention. Fully masked rows produce zero outputs and
gradients. Additive masks support gradients, including broadcast masks.

Work is partitioned across batch/head pairs. When there are fewer pairs than
twice the GPU count and they do not divide evenly across GPUs, noncausal
attention without dropout and with at least 256 key positions splits the K/V
sequence into tiles of 64 rows. Each GPU processes all query rows and heads
against its K/V slice. Partial outputs are combined using their log-sum-exp
normalizers. Backward writes partial query gradients into device-local buffers,
combines them with NCCL reduce-scatter, and copies each reduced slice into the
managed result. Key/value gradients remain disjoint. This supports masks,
strided and broadcast inputs, and deterministic backward.

Otherwise, imbalanced noncausal attention with at least 256 query positions
splits the query sequence into tiles of 64 rows and combines partial key/value
gradients in backward. This path also supports dropout. Causal attention and
shorter sequences retain whole batch/head partitions.

The fused path requires matching batch/head counts, Q/K and value channel
dimensions divisible by four and at most 65,536, `dropout_p` from 0 to 1, and
supported Ampere-or-newer GPUs. Other configurations use the math backend when enabled.

## Supported Operators

Foreach optimizer operations have multi-tensor DGX kernels for FP16, BF16,
FP32, and FP64: add, subtract, multiply, divide, maximum, lerp, addcmul,
addcdiv, sqrt, reciprocal, and negation, with in-place and out-of-place
variants. Scalar, scalar-list, and tensor-list overloads are supported where
applicable, including scalar-tensor add/multiply/divide and packed CPU scalar
tensors for addcmul/addcdiv. Dense tensors with matching layouts are batched
across GPUs; broadcasting, mixed dtypes, unsupported layouts, and potentially
dependent in-place aliases use PyTorch's existing per-tensor implementations.
The kernels preserve autograd version tracking and serve foreach Adam,
AdamW, and SGD training steps.

Log-softmax supports forward and backward for FP16, BF16, FP32, and FP64
tensors, including strided inputs and outputs, any normalization dimension,
and FP16-to-FP32 output promotion.

Tanh supports forward and backward for FP16, BF16, FP32, and FP64 tensors,
including strided inputs and outputs.

Hypot supports FP16, BF16, FP32, and FP64 inputs, broadcasting, CPU scalar
operands, mixed dtypes, and strided inputs and outputs. The functional,
in-place, and `out=` variants use the DGX kernel.

Argmax and argmin support FP16, BF16, FP32, FP64, and integer tensors, including
strided inputs and outputs, reduction over any dimension, and the flattened
reduction when no dimension is given. Ties resolve to the lowest index and NaN
outranks every other value, matching PyTorch. The reduction parallelizes over
the output elements, so cases that produce few of them, such as reducing
dimension 0 of a tall, narrow matrix or reducing a flattened input, do not fill
the GPUs.

NLL loss supports unbatched, batched, and spatial inputs, optional class
weights, `ignore_index`, and `none`, `sum`, and `mean` reductions, with forward
and backward kernels for FP16, BF16, FP32, and FP64. Mean reduction divides by
the sum of the weights of non-ignored targets. These kernels also support the
class-index cross-entropy path with `label_smoothing=0` through log-softmax and
NLL loss.

Fused Adam and AdamW support aligned, contiguous FP32 and FP64 tensors with a
scalar learning rate and FP32 step tensors on DGX. AdamW supports decoupled
weight decay; Adam requires zero weight decay. AMSGrad, maximize, and AMP
gradient scaling are not supported.

- `_copy_from_and_resize`
- `_copy_from`
- `_fused_adam_`
- `_fused_adamw_`
- `_fused_rms_norm`
- `_fused_rms_norm_backward`
- `_fused_sdp_choice`
- `_index_put_impl_`
- `_local_scalar_dense`
- `_log_softmax.out`
- `_log_softmax_backward_data.out`
- `_reshape_alias`
- `_safe_softmax`
- `_scaled_dot_product_efficient_attention`
- `_scaled_dot_product_efficient_attention_backward`
- `_softmax.out`
- `_softmax_backward_data.out`
- `abs.out`
- `add.out`
- `addcdiv.out`
- `addcmul.out`
- `addmm.out`
- `all.all_out`
- `all.out`
- `any.all_out`
- `any.out`
- `arange.start_out`
- `argmax.out`
- `argmin.out`
- `as_strided`
- `bincount`
- `bitwise_and.Tensor_out`
- `bitwise_not.out`
- `bitwise_or.Tensor_out`
- `bmm.out`
- `cat.out`
- `ceil.out`
- `clamp.out`
- `clamp_max.out`
- `clamp_min.out`
- `convolution_overrideable`
- `cos.out`
- `cumsum.out`
- `div.out`
- `embedding_dense_backward`
- `empty.memory_format`
- `empty_strided`
- `eq.Scalar_out`
- `eq.Tensor_out`
- `equal`
- `exp.out`
- `eye.m_out`
- `eye.out`
- `fill_.Scalar`
- `floor.out`
- `gather.out`
- `ge.Scalar_out`
- `ge.Tensor_out`
- `gelu.out`
- `gelu_backward.grad_input`
- `gt.Scalar_out`
- `gt.Tensor_out`
- `hypot.out`
- `index.Tensor_out`
- `index_add.out`
- `index_select`
- `isnan`
- `isneginf.out`
- `le.Scalar_out`
- `le.Tensor_out`
- `lerp.Scalar_out`
- `lerp.Tensor_out`
- `linalg_cross.out`
- `linalg_inv_ex.inverse`
- `linalg_vector_norm.out`
- `linear`
- `linear_backward`
- `log.out`
- `log10.out`
- `logical_and.out`
- `logical_or.out`
- `logit`
- `lt.Scalar_out`
- `lt.Tensor_out`
- `masked_fill_.Scalar`
- `masked_select`
- `max.dim_max`
- `max`
- `mean.out`
- `min.dim_min`
- `min`
- `mm.out`
- `mse_loss.out`
- `mse_loss_backward.grad_input`
- `mse_loss_backward`
- `mul.out`
- `multinomial`
- `native_batch_norm`
- `native_dropout`
- `native_dropout_backward`
- `native_layer_norm_backward`
- `nll_loss_forward.output`
- `nll_loss_backward.grad_input`
- `nll_loss2d_forward`
- `nll_loss2d_forward.output`
- `nll_loss2d_backward`
- `nll_loss2d_backward.grad_input`
- `ne.Scalar_out`
- `ne.Tensor_out`
- `neg.out`
- `nonzero`
- `normal_`
- `polar.out`
- `pow.Tensor_Scalar_out`
- `random_.from`
- `randperm.generator_out`
- `reciprocal.out`
- `reflection_pad2d`
- `relu_`
- `relu`
- `remainder.Tensor_out`
- `repeat_interleave.Tensor`
- `resize_`
- `round.out`
- `rsqrt.out`
- `scatter.value_out`
- `scatter_add.out`
- `set_.source_Storage_storage_offset`
- `set_.source_Storage`
- `set_.source_Tensor`
- `sgn.out`
- `sigmoid.out`
- `sigmoid_backward.grad_input`
- `sign.out`
- `silu.out`
- `sin.out`
- `sort.values_stable`
- `sqrt.out`
- `sub.out`
- `sum.IntList_out`
- `tanh.out`
- `tanh_backward.grad_input`
- `threshold_backward.grad_input`
- `tril.out`
- `triu.out`
- `unfold`
- `uniform_`
- `upsample_nearest2d.out`
- `var_mean.correction`
- `view_as_complex`
- `view_as_real`
- `view`
- `where.self`
- `where.self_out`
- `zero_`
