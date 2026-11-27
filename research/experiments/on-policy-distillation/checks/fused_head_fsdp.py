"""The fused head under FSDP2 as verl wraps an untied model (the 9B): is the lm_head's gradient right?

    torchrun --nproc_per_node=2 fused_head_fsdp.py <verl src>

A toy causal LM (embedding, one block, an untied lm_head) is wrapped as verl's apply_fsdp2 wraps one
(_select_fsdp2_wrap_targets: the block, embed_tokens and lm_head each a unit, then the root), with and without
CPUOffloadPolicy, and as the root would hold the lm_head (not a unit of its own). Its head runs as verl's fused forward
runs it (qwen3_5.py: lm_head.weight, full_tensor() if a DTensor, FusedLinearForPPO). Each rank takes different tokens
(as Ulysses ranks or data-parallel ranks do), and the loss is each rank's mean, which FSDP averages. The gradients are
compared with one process's on the same data (the mean of the ranks' gradients)."""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, sys.argv[1])
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard  # noqa: E402
from torch.distributed.tensor import DTensor  # noqa: E402

from verl.utils.experimental.torch_functional import FusedLinearForPPO  # noqa: E402

V, H, T = 4096, 256, 512


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(V, H)
        self.block = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, H))
        self.lm_head = nn.Linear(H, V, bias=False)

    def forward(self, ids):
        h = self.block(self.embed_tokens(ids))
        w = self.lm_head.weight
        if isinstance(w, DTensor):
            w = w.full_tensor()
        logp, _ = FusedLinearForPPO(chunk_size=128).forward(hidden_states=h, vocab_weights=w, input_ids=torch.roll(ids, -1), temperature=1.0)
        return -logp.mean()


def grads(model):
    out = {}
    for n, p in model.named_parameters():
        g = p.grad
        if isinstance(g, DTensor):
            g = g.full_tensor()
        out[n] = g.detach().float().cpu()
    return out


def main():
    dist.init_process_group("cpu:gloo,cuda:nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    torch.manual_seed(0)
    ids = torch.randint(0, V, (world, T), device="cuda")
    reference = Toy().cuda()
    state = {k: v.clone() for k, v in reference.state_dict().items()}
    for r in range(world):  # the ranks' mean loss, in one process
        (reference(ids[r]) / world).backward()
    want = grads(reference)

    for name, separate, offload in [("verl: lm_head a unit, on GPU", True, False), ("verl: lm_head a unit, CPU offload", True, True),
                                    ("lm_head in the root, CPU offload", False, True), ("lm_head in the root, on GPU", False, False)]:
        model = Toy().cuda()
        model.load_state_dict(state)
        kw = {"offload_policy": CPUOffloadPolicy(pin_memory=True)} if offload else {}
        units = [model.block, model.embed_tokens] + ([model.lm_head] if separate else [])
        for m in units:
            fully_shard(m, **kw)
        fully_shard(model, **kw)
        try:
            model(ids[rank]).backward()
            got = grads(model)
            err = {n: ((got[n] - want[n]).norm() / want[n].norm()).item() for n in want}
            verdict = "OK" if max(err.values()) < 1e-3 else "WRONG"
            msg = f"{verdict} rel. grad error " + ", ".join(f"{n} {e:.1e}" for n, e in err.items())
        except Exception as e:  # noqa: BLE001
            msg = f"FAILS {type(e).__name__}: {str(e)[:150]}"
        if rank == 0:
            print(f"{name}: {msg}", flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    main()
