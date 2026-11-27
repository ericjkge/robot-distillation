"""SFT's fused head (verl's FusedLinearForPPO, torch backend, as SFT 60955095 runs it) against the plain head on one GPU:
log-probs and the gradients of the hidden states and the head's weights, at Qwen3.5-9B's width and vocabulary."""

import sys

import torch

sys.path.insert(0, sys.argv[1])  # verl's src
from verl.utils.experimental.torch_functional import FusedLinearForPPO  # noqa: E402

torch.manual_seed(0)
T, H, V = 3000, 4096, 248320
dev = "cuda"
h = (torch.randn(T, H, device=dev) / 4).to(torch.bfloat16).requires_grad_()
w = (torch.randn(V, H, device=dev) / 64).to(torch.bfloat16).requires_grad_()
y = torch.randint(0, V, (T,), device=dev)
c = torch.randn(T, device=dev)
torch.cuda.reset_peak_memory_stats()
lp, ent = FusedLinearForPPO(impl_backend="torch").forward(hidden_states=h, vocab_weights=w, input_ids=y, temperature=1.0)
(lp * c).sum().backward()
peak = torch.cuda.max_memory_allocated() / 2**30
gh, gw = h.grad.float(), w.grad.float()
h.grad = w.grad = None
logits = (h.float() @ w.float().T)
ref = torch.log_softmax(logits, -1).gather(-1, y[:, None])[:, 0]
(ref * c).sum().backward()
rel = lambda a, b: ((a - b).norm() / b.norm()).item()
print(f"fused peak {peak:.1f} GiB (plain logits alone {T * V * 4 / 2**30:.1f} GiB)")
print(f"log-prob max err {(lp - ref).abs().max().item():.2e}, grad_h rel err {rel(gh, h.grad.float()):.2e}, "
      f"grad_w rel err {rel(gw, w.grad.float()):.2e}")
ok = (lp - ref).abs().max().item() < 5e-2 and rel(gh, h.grad.float()) < 2e-2 and rel(gw, w.grad.float()) < 2e-2
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
