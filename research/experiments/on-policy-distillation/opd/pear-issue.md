Hi, thanks for releasing PEAR. I'm running on-policy distillation with a Qwen3.5-4B student, and one run crashed in the first actor update:

```
File "verl/workers/engine_workers.py", line 290, in train_mini_batch
File "verl/utils/tensordict_utils.py", line 610, in get_data
File "verl/utils/tensordict_utils.py", line 495, in index_select_tensor_dict
...
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 5129 (input tensor's size at dimension 1), but got split_sizes=[4, 4]
```

Qwen3.5's `position_ids` are a nested tensor of shape (batch, 4, ragged length), so `_ragged_idx` should be 2. The error means it was 1 when `make_iterator` selected the mini-batch, so torch split the 5129 tokens using the 4 position rows as lengths. `train_mini_batch` calls `maybe_fix_3d_position_ids` before building the iterator, but the iterator selects lazily, so a reset in between goes uncaught. I didn't find what resets it.

Repairing right before each selection avoids this:

```diff
         for _ in range(epochs):
             for idx in train_dataloader:
+                maybe_fix_3d_position_ids(tensordict)
                 yield index_select_tensor_dict(tensordict, idx)
```

With it, the same setup ran both updates, export and evaluation cleanly. One caveat: in that run `_ragged_idx` was already 2 before the repair every time, so the crash didn't come back and I can't prove this change is what prevents it. The repair is a no-op when the index is right, but the real cause may be earlier. Patch: https://github.com/ericjkge/robot-distillation/blob/main/patches/verl/0044-utils-fix-make_iterator-repairs-position_ids-ragged-dimension-before-every-mini-batch-selection.patch

Setup: PEAR faff474 (verl 8e03c039 + your 43 patches), `pear.main_training` with `distillation.enabled=true`, Qwen3.5-4B as student and teacher, `separate_async`, `train_batch_size=2`, `ppo_mini_batch_size=2`, 4x L40S with FlashAttention-2 and `fsdp_config.offload_policy=true`.
