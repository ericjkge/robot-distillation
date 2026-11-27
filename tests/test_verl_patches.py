"""This repository's verl patches (patches/verl), in the checkout every step runs (scripts/verl.sh)."""

import ast
from types import SimpleNamespace

import torch.nn as nn
from conftest import VERL_SRC


def select_fsdp2_wrap_targets():
    """verl's _select_fsdp2_wrap_targets, read from the checkout: its module imports what the tests lack."""
    source = (VERL_SRC / "verl" / "utils" / "fsdp_utils.py").read_text()
    functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
    (node,) = [n for n in functions if n.name == "_select_fsdp2_wrap_targets"]
    namespace = {"nn": nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "fsdp_utils.py", "exec"), namespace)
    return namespace["_select_fsdp2_wrap_targets"]


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(4, 4)


class Toy(nn.Module):
    def __init__(self, tie: bool):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=tie)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)
        self.model.layers = nn.ModuleList([Layer(), Layer()])
        self.lm_head = nn.Linear(4, 8, bias=False)


def test_an_untied_lm_head_stays_in_the_root_unit():
    """0045: the fused forward reads lm_head.weight without calling lm_head. As an FSDP2 unit of its own the head's
    hooks never run: under CPU offload the weight stays on the CPU (the 9B's SFT failed at step 1), and on the GPU its
    gradient is wrong (a 2-GPU check: relative error 1.0; in the root unit, exact)."""
    select = select_fsdp2_wrap_targets()
    untied = Toy(tie=False)
    targets = select(untied, ["Layer"])
    assert untied.lm_head not in targets
    assert untied.model.embed_tokens in targets and all(layer in targets for layer in untied.model.layers)
    tied = Toy(tie=True)  # shared with the embedding: both in the root unit, as before
    assert tied.lm_head not in select(tied, ["Layer"]) and tied.model.embed_tokens not in select(tied, ["Layer"])
