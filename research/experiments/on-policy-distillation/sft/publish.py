"""Upload an SFT checkpoint (its huggingface/ directory) to the Hub as a public model, with a short card, and add it to
the project's collection next to the teacher rollouts.
    python publish.py <checkpoint>/huggingface <repo> <collection slug> <episodes in the SFT set>"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

checkpoint, repo, collection, episodes = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
api = HfApi()
api.create_repo(repo, private=False, exist_ok=True)
card = f"""---
base_model: Qwen/Qwen3.5-9B
datasets: [eddykang06/qwen3.8-27b-pear-libero-rollouts]
tags: [robotics, libero, distillation, sft]
---

# Qwen3.5-9B, SFT on PEAR's teacher rollouts on LIBERO

[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) fine-tuned on {episodes} successful episodes of
[Qwen3.8-27B-PEAR-LIBERO_only](https://huggingface.co/mit-han-lab/Qwen3.8-27B-PEAR-LIBERO_only) at xhigh effort
([the rollouts](https://huggingface.co/datasets/eddykang06/qwen3.8-27b-pear-libero-rollouts)), up to 25 per task, with
the loss on every generated token (reasoning and actions), in the same multi-round context as at inference. lr 1e-5
cosine, 5% warmup, batch 32, 3 epochs. It uses the teacher's chat template, which sets the reasoning effort in the
system prompt. The starting point of on-policy distillation in
[robot-distillation](https://github.com/ericjkge/robot-distillation); checkpoint `{checkpoint.parent.name}`.
"""
api.upload_folder(repo_id=repo, folder_path=checkpoint, commit_message=f"SFT checkpoint {checkpoint.parent.name}")
api.upload_file(repo_id=repo, path_or_fileobj=card.encode(), path_in_repo="README.md")
api.add_collection_item(collection, item_id=repo, item_type="model", exists_ok=True)
print(f"{checkpoint} -> https://huggingface.co/{repo}, in collection {collection}")
