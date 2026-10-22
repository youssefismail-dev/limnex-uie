"""Skips the exact-bit pins unless UIE_BITEXACT=1 (see the `reference_machine` marker in pyproject.toml).

Those tests compare against bit patterns recorded on the reference machine: a sha256 of trained
weights, or strings sampled from an untrained toy model whose logits tie to within one float32
step. Floating-point results differ at the last bit between CPUs, kernel paths, thread counts and
operating systems, so on any other machine they can fail with the code unchanged.
"""
import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("UIE_BITEXACT") == "1":
        return
    skip = pytest.mark.skip(
        reason="exact-bit pin recorded on the reference machine; set UIE_BITEXACT=1 there to run it")
    for item in items:
        if "reference_machine" in item.keywords:
            item.add_marker(skip)
