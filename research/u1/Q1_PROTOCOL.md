# U1 Q1: the generation protocol

- **What this is:** the implementation choices for Q1 (`PREREG_U1.md` §§2–3) that the registration
  left open.
- **Fixed before any U1 generator is trained or run.** Code: `research/u1/q1_common.py`,
  `q1_generate.py`, `make_q1_notebooks.py`. Any change after a generator runs is recorded here as a
  dated amendment.
- **Q1 touches no test set.** It reads only UIE's training data (to train or prompt the generators)
  and, at scoring, the same real validation pools E3 used.

## 1. Common to every generator

| | rule |
|---|---|
| **training / prompt data** (§2) | **benign:** `bundle2` `name_train` + `structure_train` + E15's stream (`qry_extra.jsonl`, 57,292 rows), the rows `e16_b5` trained on. **phishing:** `phishing_name_train` + `phishing_structure_train`. Rows as stored (no de-duplication), the scheme dropped (UIE's `include_scheme=False`). |
| **output form** | scheme-less URLs, as UIE writes them (E3's FULL detector strips the scheme from real URLs too) |
| **the gate** | `uie.gate.Gate(known_membership=` the published known-URL filter, content sha256 `3ef34d06…`): validity, duplicates and copies of training URLs. A fresh gate for each generation file, applied with `reset=False`, so duplicates are removed across that file's chunks. |
| **file** | generate in chunks; keep URLs in order as they pass the gate, until **8,000**. A file whose generator cannot reach 8,000 within **200,000** raw URLs is recorded as short, and scored as it is. |
| **per-file record** | raw count, gate removals by reason, kept count, chunk seeds, seconds. **validity** = 1 − (rejections for a validity reason) / raw. **copy rate** = `memorized` / raw. |
| **layout** | `data/u1/q1/gen/free_{gen}_{label}_g{g}.json`, g = 0…4, with `PROVENANCE_{gen}.json` beside them: E12's `free_files` layout, so E12's `part_free` scores them unchanged |
| **seeds** | chunk k of generation g of class c: `base + 100_000·c + 1_000·g + k`, where c = 0 for benign and 1 for phishing. Bases: G0 `30_000_000` (§2), B1 `31_000_000`, B2 `32_000_000`, B3 `33_000_000`, B4 `34_000_000`. Training seeds: B2 `32_999_999`; B3's example draws use the chunk seed. |

**Generator ids:** `uie` (G0), `ngram` (B1), `lstm` (B2), `llm` (B3), `twist` (B4).

## 2. The generators

### G0, `uie`
- The published `e16_b5`, run with `uie.engine.generate_free` and the public interface's own
  arguments (its shipped suffix table and presence-slash table, E16_B5's decode).
- The result passes through the common gate. This is exactly what `Engine.generate` does
  (`uie/api.py`), done in one place so every generator meets the same gate.
- Chunks of 2,000. Runs on a Kaggle GPU.

### B1, `ngram`
- `uie.ngram.NgramURLModel(order=5)`: E1's baseline, built from `CharNgram(order=5)` per field (label,
  subdomain, path, query, fragment), with categorical suffix, subdomain count and shape.
- Fitted per class on that class's training rows. Chunks of 4,000. Kaggle (CPU work, in the G0
  notebook).

### B2, `lstm`
| | |
|---|---|
| **architecture** | embedding 128 → 2-layer LSTM, 512 units, dropout 0.1 → linear over UIE's vocabulary |
| **tokens** | `e16_b5`'s own tokenizer (110 symbols, exported to `uie_tokenizer_e16_b5.json`), `encode(url, label, frame)`, with frame `domain` for name rows and `page` for structure and stream rows, as UIE trains. Sequences are capped at 384 tokens (UIE's context). The class token, frame and plan tokens are ordinary targets: the LSTM learns them and samples them freely. |
| **data** | the two classes' training rows pooled at their natural ratio, shuffled with the training seed |
| **training** | AdamW, learning rate 2e-3, batch 256, gradient clip 1.0, plain next-token loss on every token (no field masking). It stops at **2 GPU-hours of training wall time** on one T4 (§2's budget). Validation loss on 2,000 val rows per class is logged and never used to select. |
| **sampling** | temperature 1.0, from `<BOS> <C:label>`, up to 384 tokens, decoded with `URLTokenizer.decode`. Chunks of 2,000. |

### B3, `llm`
| | |
|---|---|
| **model** | `Qwen/Qwen2.5-7B-Instruct` at the F1 pin `a09a35458c702b33eeacc393d103063234e8bc28`, float16. vLLM with tensor parallel 2 on Kaggle's 2×T4, or, if vLLM cannot be installed, `transformers` with `device_map="auto"` (same model, prompt and sampling). The backend used is recorded. |
| **prompt** | Appendix A exactly, as plain text (the appendix's bold and code marks are formatting), as the user turn of the model's chat template: `{label}` = `benign` / `phishing-style`, `{k}` = 20. The 20 examples are drawn uniformly from that class's training rows with the chunk seed, scheme dropped. Each prompt's sampling seed is `chunk seed × 1000 + i`. |
| **sampling** | temperature 1.0, top-p 1.0, max 1,024 new tokens, one sample per prompt, 64 prompts per chunk |
| **parsing** | each response line is stripped of whitespace, list markers (`-`, `*`, `•`, `1.`, `1)`), surrounding backticks or quotes, and a scheme. A line becomes a candidate if it then contains a dot and no space. |
| **refusal** | a response with no candidate that passes the gate's validity checks. Refusals are counted per file. |

### B4, `twist` (phishing only; excluded from Q2, §2)
| | |
|---|---|
| **host** | a benign training registrable domain, drawn uniformly by row, with **one** dnstwist-style edit chosen uniformly from: homoglyph (ASCII only: `o↔0`, `l↔1`, `i↔l`, `m→rn`, `w→vv`, `e→3`, `a→4`, `s→5`), insertion (a random letter or digit), omission, transposition of two adjacent characters, hyphenation (a `-` at a random inner position), and suffix swap (one of the 20 most common phishing training suffixes). |
| **path** | a *template* is a phishing training URL's path + `?query` with every digit replaced by `0`. The empty path counts as a template. The 500 most frequent templates are sampled in proportion to their training counts, and every `0` is filled with a random digit. |
| **ASCII only** | UIE's 110-symbol alphabet and the real pools are ASCII-dominated, so Unicode homoglyphs would be a trivial tell rather than a test |
| **where** | it has no training, so it runs on the laptop (`q1_generate.py twist`). Chunks of 4,000. |

### Note (2026-10-21, before any B3 output existed): B3 runs on `transformers`
The first B3 run installed vLLM. That replaced Kaggle's PyTorch (2.11, CUDA 12.8) with 2.13 (CUDA 13.0)
but kept torchaudio at CUDA 12.8, so both backends failed to import. No URL was generated.
- B3 now uses the registered fallback, `transformers` with `device_map="auto"`, on Kaggle's own
  PyTorch: the same model, revision, prompt and sampling.
- vLLM is tried only on request (`U1_VLLM=1`).
- Each `generate()` call takes the chunk's 64 prompts at once. That changes speed, not the
  sampling distribution.

## 3. Scoring (after every generator's files exist)
- E12's `part_free` instrument, unchanged, with `bundle2`'s validation pools, per generator and class
  (§3). B4 is scored for phishing only.
- **Also reported:** validity and copy rate (from the provenance), distinct registrable domains per
  8,000, suffix entropy, and E18's word tell (informational).
- **The readout is §3's:** x(UIE) − x(B), Welch intervals over 5 + 5 generation means, one-sided 95%.
