"""Build research/d2/reproduce_gate.ipynb, the Kaggle notebook for REPRODUCE.md level 2b.

    python research/d2/make_reproduce_notebook.py

The notebook only runs `research/d2/reproduce_gate.py generate` on Kaggle's GPU: everything that
matters is in that script, which is tested and identical on any machine.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "reproduce_gate.ipynb"

INTRO = """# Limnex UIE — reproduce the E3 gate's 50 files (REPRODUCE.md level 2b)

Regenerates the 50 files the E3 gate read (180,000 URLs) with the published model, and checks each
one against the published fingerprint as it is written. About 2.5 hours.

**Before you run:** Settings → Accelerator → **GPU T4**; Settings → **Internet on**.
Then **Run all**. When it finishes, download `repro_gen.zip` from the output and unzip it into
your clone's `data/repro/` (giving `data/repro/gen/`), then run level 2c on your computer.

The files include synthetic phishing-style URLs. Read the README's **Responsible Use** and
**No Liability** sections first: never open, register, resolve, host or send them.

The code and the model come from GitHub. To use your own copies instead, attach a Kaggle dataset
holding the repository folder and `limnex-uie-e16_b5.pt`; the first cell finds them."""

SETUP = """import pathlib, subprocess, sys
WORK, INPUT = pathlib.Path("/kaggle/working"), pathlib.Path("/kaggle/input")
REPO_URL = "https://github.com/youssefismail-dev/limnex-uie"
MODEL_URL = REPO_URL + "/releases/download/v0.1.0/limnex-uie-e16_b5.pt"

# an attached dataset wins; otherwise GitHub
repo = next((p.parent for p in INPUT.rglob("pyproject.toml") if (p.parent / "uie" / "api.py").is_file()), None)
model = next(iter(INPUT.rglob("limnex-uie-e16_b5.pt")), None)
if repo is None:
    subprocess.run(["git", "clone", "--quiet", "--depth", "1", REPO_URL + ".git", str(WORK / "limnex-uie")], check=True)
    repo = WORK / "limnex-uie"
if model is None:
    subprocess.run(["curl", "-sSL", "-o", str(WORK / "limnex-uie-e16_b5.pt"), MODEL_URL], check=True)
    model = WORK / "limnex-uie-e16_b5.pt"
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--progress-bar", "off", "tldextract==5.3.1"], check=True)
import torch
assert torch.cuda.is_available(), "No GPU: Settings -> Accelerator -> GPU T4, then run again"
print("repository:", repo)
print("model:", model)
print("gpu:", torch.cuda.get_device_name(0), "| torch", torch.__version__)"""

GENERATE = """# writes /kaggle/working/gen; a rerun keeps every file that already matches. The script's output
# is read here and printed once (left to itself, Kaggle's log shows a subprocess's lines twice).
p = subprocess.Popen([sys.executable, "-u", str(repo / "research" / "d2" / "reproduce_gate.py"), "generate",
                      "--weights", str(model), "--device", "cuda", "--out", str(WORK / "gen")],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="", flush=True)
if p.wait() != 0:
    raise SystemExit(f"FAILED (exit {p.returncode}): not all 50 files were written and matched -- see the messages above")
print("OK: all 50 files identical to the published run")"""

PACK = """import shutil
shutil.make_archive(str(WORK / "repro_gen"), "zip", str(WORK), "gen")
print("download", WORK / "repro_gen.zip", "and unzip it into data/repro/ in your clone")"""


def cell(kind: str, text: str) -> dict:
    lines = text.split("\n")
    src = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


nb = {"cells": [cell("markdown", INTRO), cell("code", SETUP), cell("code", GENERATE), cell("code", PACK)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT)
