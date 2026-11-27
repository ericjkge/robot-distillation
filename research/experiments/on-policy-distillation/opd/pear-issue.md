Hi, thanks for releasing PEAR. I'm running on-policy distillation (`distillation.enabled=true`) with a Qwen3.5-4B student and hit a crash in the actor update. The first update goes through, then a later mini-batch selection fails:

```
File "verl/workers/engine_workers.py", line 290, in train_mini_batch
File "verl/utils/tensordict_utils.py", line 610, in get_data
File "verl/utils/tensordict_utils.py", line 495, in index_select_tensor_dict
...
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 5129 (input tensor's size at dimension 1), but got split_sizes=[4, 4]
```

From what I can tell, Qwen3.5's `position_ids` is a nested tensor of shape (batch, 4, ragged length), so its ragged dim should be 2. By the time `make_iterator` picks a mini-batch, `_ragged_idx` has become 1, and the split uses the 4 position rows as lengths. `train_mini_batch` does call `maybe_fix_3d_position_ids` before building the iterator, but the iterator selects lazily and the index gets reset somewhere in between. I couldn't find where; on CPU the fix survives a pickle round trip.

Calling the fix right before each selection solved it for us:

```diff
         for _ in range(epochs):
             for idx in train_dataloader:
+                maybe_fix_3d_position_ids(tensordict)
                 yield index_select_tensor_dict(tensordict, idx)
```

With that change the run gets through both updates, and exporting and evaluating the result work too. Here it is as a patch in your format, in case it's useful: https://github.com/ericjkge/robot-distillation/blob/main/patches/verl/0044-utils-fix-make_iterator-repairs-position_ids-ragged-dimension-before-every-mini-batch-selection.patch

Setup: PEAR at faff474 (verl 8e03c039 + your 43 patches), `pear.main_training` with distillation, `trainer_mode: separate_async`, Qwen3.5-4B as both student and teacher, `train_batch_size=2`, `ppo_mini_batch_size=2`, on 4x L40S (FlashAttention-2 instead of your kernels, `fsdp_config.offload_policy=true`).
