Hi! Thanks for releasing PEAR. I'm using it for on-policy distillation with a Qwen3.5-4B student, and training crashes after the first update. I found the cause and a one-line fix, details below.

**What I ran:** `pear.main_training` with `distillation.enabled=true`, Qwen3.5-4B as student and teacher, `trainer_mode: separate_async`, `train_batch_size=2`, `ppo_mini_batch_size=2`. PEAR at faff474 (verl 8e03c039 plus your 43 patches), on 4x L40S with FlashAttention-2 instead of your custom kernels and `fsdp_config.offload_policy=true`.

**What happens:** the first update finishes normally. Then the actor update crashes while verl selects the next mini-batch:

```
File "verl/workers/engine_workers.py", line 290, in train_mini_batch
File "verl/utils/tensordict_utils.py", line 610, in get_data
File "verl/utils/tensordict_utils.py", line 495, in index_select_tensor_dict
...
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 5129 (input tensor's size at dimension 1), but got split_sizes=[4, 4]
```

**Why it happens:** for Qwen3.5, each sample's `position_ids` has shape (4, sequence length), and the batch stores them as a nested tensor of shape (batch, 4, ragged length). The ragged dimension is dim 2, and the nested tensor records that in `_ragged_idx`. When the crash happens, `_ragged_idx` is 1 instead of 2. So when `index_select_tensor_dict` splits the batch into samples, it takes the lengths from the wrong dimension: it tries to cut the 5129 packed tokens into pieces of 4 and 4 (the number of position rows per sample), which can't work.

verl already has a function that repairs this, `maybe_fix_3d_position_ids`, and `train_mini_batch` calls it on the batch before creating the mini-batch iterator. But `make_iterator` selects mini-batches lazily, one at a time, and by the time it selects one, `_ragged_idx` is back to 1. I couldn't find what resets it. On CPU, the repaired tensor keeps `_ragged_idx == 2` even through a pickle round trip, so it seems to happen somewhere on the GPU path.

**The fix:** call the repair inside the iterator, right before every selection (`verl/utils/tensordict_utils.py`, in `make_iterator`):

```diff
         for _ in range(epochs):
             for idx in train_dataloader:
+                maybe_fix_3d_position_ids(tensordict)
                 yield index_select_tensor_dict(tensordict, idx)
```

With this change, the same run gets through both updates, and exporting and evaluating the trained model work too. I also logged `_ragged_idx` right before each selection, and it was 2 every time. The same change as a patch in your patch format: https://github.com/ericjkge/robot-distillation/blob/main/patches/verl/0044-utils-fix-make_iterator-repairs-position_ids-ragged-dimension-before-every-mini-batch-selection.patch

Happy to test anything else if that helps.
