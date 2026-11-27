Title: Training a Qwen3.5 student fails at the second mini-batch selection: position_ids lose their ragged dimension (split_with_sizes error)

### What happens

On-policy distillation (`distillation.enabled=true`) with a Qwen3.5-4B student fails inside the actor update, when
verl selects a mini-batch from the batch:

```
File "verl/workers/engine_workers.py", line 290, in train_mini_batch
File "verl/utils/tensordict_utils.py", line 610, in get_data
File "verl/utils/tensordict_utils.py", line 495, in index_select_tensor_dict
...
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 5129 (input tensor's size at dimension 1), but got split_sizes=[4, 4]
```

The first update succeeds. The failure comes on a later selection.

### Why

Qwen3.5's `position_ids` are a nested tensor of shape (batch, 4, ragged length), with the ragged dimension at index 2.
By the time `make_iterator` selects a mini-batch, `position_ids._ragged_idx` is 1, so the selection reads the
per-sample lengths off the wrong dimension (4, the position rows) and splits the 5129 tokens by them.

`train_mini_batch` calls `maybe_fix_3d_position_ids` on the batch before it builds the iterator, but `make_iterator`
selects lazily, and by then the index has been reset. We did not find which step resets it: on CPU, the repair survives
a pickle round trip through the transfer queue's format.

### Fix

Repair the batch right before each selection (`verl/utils/tensordict_utils.py`, `make_iterator`):

```diff
         for _ in range(epochs):
             for idx in train_dataloader:
+                maybe_fix_3d_position_ids(tensordict)
                 yield index_select_tensor_dict(tensordict, idx)
```

With this, the same run completes both updates, exports the model and evaluates it. A probe printed `_ragged_idx == 2`
before every selection. The change as a patch in PEAR's format:
https://github.com/ericjkge/robot-distillation/blob/main/patches/verl/0044-utils-fix-make_iterator-repairs-position_ids-ragged-dimension-before-every-mini-batch-selection.patch

### Setup

- PEAR at faff474 (verl 8e03c039 with PEAR's 43 patches)
- `pear.main_training` with on-policy distillation, `trainer_mode: separate_async`
- Student Qwen3.5-4B, teacher Qwen3.5-4B, `data.train_batch_size=2`, `actor_rollout_ref.actor.ppo_mini_batch_size=2`
- 4× L40S: FlashAttention-2 in place of PEAR's kernels, `actor_rollout_ref.actor.fsdp_config.offload_policy=true`
