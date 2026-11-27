Hi, thanks for releasing PEAR. I'm using it for on-policy distillation with a Qwen3.5-4B student, and one of my runs crashed in the first actor update with this:

```
File "verl/workers/engine_workers.py", line 290, in train_mini_batch
File "verl/utils/tensordict_utils.py", line 610, in get_data
File "verl/utils/tensordict_utils.py", line 495, in index_select_tensor_dict
...
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 5129 (input tensor's size at dimension 1), but got split_sizes=[4, 4]
```

As far as I can tell, this is what goes wrong. For Qwen3.5, each sample's `position_ids` has shape (4, sequence length), and the batch holds them as a nested tensor of shape (batch, 4, ragged length). The ragged dimension is dim 2, which the nested tensor stores in `_ragged_idx`. The error means `_ragged_idx` was 1 when `make_iterator` selected the mini-batch: torch took the sample lengths from the wrong dimension and tried to cut the 5129 packed tokens into pieces of 4 and 4, the number of position rows.

verl has a function that repairs this, `maybe_fix_3d_position_ids`, and `train_mini_batch` calls it before it creates the iterator. But the iterator selects lazily, so if anything resets the index between that call and a selection, nothing catches it. I didn't find what reset it.

I added the repair right before each selection in `make_iterator` (`verl/utils/tensordict_utils.py`):

```diff
         for _ in range(epochs):
             for idx in train_dataloader:
+                maybe_fix_3d_position_ids(tensordict)
                 yield index_select_tensor_dict(tensordict, idx)
```

The same setup then ran both updates and the export and evaluation without errors. To be upfront, I also logged `_ragged_idx` just before the repair in that run and it was already 2 every time, so the bad state didn't come back and I can't say for certain this change is what prevented the crash. The repair costs nothing when the index is already right, so it seemed like the safe place for it, but there may be an earlier root cause worth finding. Here's the change in your patch format: https://github.com/ericjkge/robot-distillation/blob/main/patches/verl/0044-utils-fix-make_iterator-repairs-position_ids-ragged-dimension-before-every-mini-batch-selection.patch

My setup: PEAR at faff474 (verl 8e03c039 with your 43 patches), `pear.main_training` with `distillation.enabled=true`, Qwen3.5-4B as both student and teacher, `trainer_mode: separate_async`, `train_batch_size=2`, `ppo_mini_batch_size=2`, on 4x L40S with FlashAttention-2 instead of your custom kernels and `fsdp_config.offload_policy=true`.

Happy to run anything that would help narrow it down.
