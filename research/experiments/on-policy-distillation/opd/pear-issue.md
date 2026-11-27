**Title:** verl `make_iterator`: mini-batch selection fails on Qwen3.5's nested 3D `position_ids` after the transfer queue (`split_with_sizes ... got split_sizes=[4, 4]`)

**Setup.** PEAR at faff474 (verl 8e03c039 + PEAR's 43 patches), `pear.main_training` with on-policy distillation
(`distillation.enabled=true`), student Qwen3.5-4B, `trainer_mode: separate_async`, generic GPUs (4× L40S / A6000;
FlashAttention-2, actor `fsdp_config.offload_policy=true`), `data.train_batch_size=2`, `ppo_mini_batch_size=2`.

**Symptom.** The first update completes; the actor's update then fails while selecting a mini-batch:

```
verl/workers/engine_workers.py", line 290, in train_mini_batch
verl/utils/tensordict_utils.py", line 610, in get_data
verl/utils/tensordict_utils.py", line 495, in index_select_tensor_dict
torch/nested/_internal/ops.py", line 1170, in unbind_int
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 5129 (input tensor's size at dimension 1), but got split_sizes=[4, 4]
```

**Cause, as far as we traced it.** Qwen3.5's `position_ids` are a nested tensor of shape (batch, 3 or 4, ragged seq)
whose ragged dimension is 2. After the batch passes through the transfer queue, the nested tensor's `_ragged_idx`
can read 1, so `index_select_tensor_dict` splits along the wrong dimension. `maybe_fix_3d_position_ids` repairs it, and
`train_mini_batch` calls it on the batch before building the iterator. On CPU that repair survives a consolidated
pickle round trip, so something on the GPU path undoes it before the selections that `make_iterator` performs
lazily; we did not pin down what.

**Fix that works for us.** Repair the batch before every selection in `make_iterator`
(`verl/utils/tensordict_utils.py`):

```diff
         for _ in range(epochs):
             for idx in train_dataloader:
+                # the selection reads position_ids' ragged dimension, which a transfer through PEAR's queue resets
+                maybe_fix_3d_position_ids(tensordict)
                 yield index_select_tensor_dict(tensordict, idx)
```

With it, a 2-update distillation run completes on 4× L40S (a probe printed `_ragged_idx == 2` before every
selection), and the export and evaluation of the result run. The patch, in PEAR's format, is
`patches/verl/0044-utils-fix-make_iterator-repairs-position_ids-ragged-dimension-before-every-mini-batch-selection.patch`
in https://github.com/ericjkge/robot-distillation.
