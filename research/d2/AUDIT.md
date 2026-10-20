# D2 release audit — report

Plan: `D2_PLAN.md`, steps 1 and 2 (decisions locked 2026-10-17). Scanner: `audit.py` (re-runnable);
machine-readable result: `AUDIT.json` (locations and categories only — no matched URL, domain or
secret is written anywhere). Run 2026-10-17.

## In plain words

- **The files that will become the public repository are clean.** No real phishing URL, host or
  domain; no secret; no local path, username or e-mail address. What the scanner still lists is a
  set of reviewed false alarms, each with its reason below.
- **No secret was ever committed** — not now, not in any of the 1,008 file versions in history.
- **This repository's history must stay private.** Old versions hold local paths, the username, the
  corpus's domain lists and a few real phishing-associated domain names. That is why the public
  release is a fresh repository with one starting commit (decision C).

## What was scanned, against what

| scope | what |
|---|---|
| current files | the 595 files the next commit would hold (tracked + new, not ignored), `.gz` decompressed |
| history | all 1,008 file versions ever committed, on every branch |
| real phishing | 322,552 phishing URLs, 240,918 hosts and 93,997 registrable domains from the project's own corpus (`data/corpus`, `corpus2`, `bundle2`); a host or domain counts only if it never occurs in the benign corpus, so shared platforms are not flagged by themselves |
| secrets | Kaggle, GitHub, ngrok, AWS, Google, OpenAI, Anthropic and Hugging Face token formats; private keys; assigned passwords and tokens |
| private details | local paths in every style (`E:\…`, `/e/…`, `/home/…`, `/Users/…`), this machine's username, e-mail addresses, machine names |
| brands | hostnames containing a well-known brand inside Markdown documents |
| bulk lists | any file holding 50 or more distinct real hostnames |

## Found and fixed (current files)

| finding | fix |
|---|---|
| **Two tracked domain lists** — `data/corpus/split_random.json.gz` and `split_temporal.json.gz` hold all 193,670 registrable domains of the corpus, real phishing domains among them (an earlier note said they "redistribute nothing"; the audit showed otherwise) | **untracked** (kept on disk and in the private history). Their manifests — rule, seed, counts and `content_sha256`, no domains — are tracked in `data/corpus/splits.manifest.json`; a split rebuilt from the recipe is deterministic and still hash-verified. `.gitignore` records why |
| **Real brands in hand-written phishing-style examples** (README, DESIGN: two banks; one fake of a payment brand with a look-alike letter) | replaced with fictional brands (`examplebank`, `examplecredit`, and `exampIebank` — the look-alike capital I is kept, so the example still shows the trick) |
| **Real brands in code comments and tests** — 29 strings in 15 files (the `@`-trick example, fake bank and payment hosts) | replaced with fictional names of the **same length** (`brandx`, `bankx`, `trust`, `examp1e`), so tests that compare character fragments behave as before; the eleven affected test files pass |
| **One real domain named in an analysis** (`research/v2/PREREG_V2_P2.md`: a `.com.au` domain counted 13,351 times in the phishing feed) | described without its name; the number is unchanged |
| **Local paths** (step 1) — every hard-coded local path, in 50 files (45 Python scripts, 4 shell scripts, 1 Markdown file), including Git-Bash-style drive paths the first count missed | repository-relative paths; 76 repository paths proved to point at the same file as before; the old scratch folder became the git-ignored `data/scratch/`; all 294 research scripts compile |
| **The username inside the audit script itself** (its own search pattern) | read from the machine at run time; no name is written in the file |

## Reviewed and kept (not findings)

| category | where | why it stays |
|---|---|---|
| hosting platforms named in analysis | 8 files (`vercel.app`, `pages.dev`, `netlify.app`, `github.io`, `blogspot.com`, `appspot.com`, `fc2.com`, `weebly.com`, `godaddysites.com`, …) | the domain belongs to the platform, which hosts benign and phishing sites alike; naming it points at no one's site |
| `user@host` URLs | 10 files (tests and code comments: `http://user:pass@…`, the `http://brandx.com@evil.tk/` trick) | URL syntax that the parser is tested on, not e-mail addresses |
| model output with brand names | `E3_REPORT.md` §3.4 qualitative samples and its parse example | what the model generated; none matches any corpus URL or host; changing it would falsify the evidence |
| random strings | `tests/fixtures_e7_generate.json`, `tests/fixtures_e5_path.json` | output of a tiny untrained test model; some happen to contain `@` or a real top-level domain |

## History (stays private — decision C)

| category | file versions | paths |
|---|---|---|
| local paths | 59 | 50 |
| username | 15 | 15 |
| real phishing hosts (the two domain lists) | 2 | 2 |
| real phishing-associated domains named | 7 | 3 |
| brand examples later replaced | 19 | 2 |
| bulk lists | 2 | 2 |
| **secrets** | **0** | — |

## Still to do before going public (outside this audit)

- **Step 3 documents:** `REPRODUCE.md` must say that the splits are rebuilt (`uie.corpus.build_splits`)
  and checked against `data/corpus/splits.manifest.json`.
- **The public repository's single commit** uses GitHub's private no-reply e-mail address.
- **Generated URLs shipped in any new document** pass the active-threat screen first (README,
  "Responsible use").
- **Legal read** of the LICENSE, README disclaimer and model card (decision A, before E).

## Honest limits of this audit

- It finds real phishing only where the project's own corpus knows it; a phishing URL from
  elsewhere would not be recognised.
- It cannot know whether a model-generated host happens to exist on the live web; that is what the
  active-threat screen at publication is for, and why no bulk generated corpus is shipped.
- It does not judge the model weights. They generate phishing-style URLs by design; the
  Responsible Use and No Liability statements (step 3) govern that.

## Re-runs

| date | after | files | result |
|---|---|---|---|
| 2026-10-17 | steps 3–4 (LICENSE, NOTICE, MODEL_CARD.md, REPRODUCE.md, README sections, `uie/api.py`, `uie/engine.py`, `uie/cli.py`, `uie/assets/`, `data/corpus/bundle2.manifest.json`) | 610 | **CLEAN** (only the reviewed exceptions above); history unchanged |
| 2026-10-17 | REPRODUCE.md opens in plain words (the model is published; the file is for checking results; the fingerprint explained) | 610 | **CLEAN**; history 1,105 blobs — the step 3–4 commits added only reviewed exceptions |
| 2026-10-17 | README "Acknowledgements" (built with the help of AI coding tools, including Claude Code); D2 decision F | 610 | **CLEAN**; history 1,109 blobs — cc79ea2 added only reviewed exceptions |
| 2026-10-17 | step 5: the filter check, the quick check (`quick_check.py`, `GATE_FINGERPRINTS.json`), the E15 pin, the release builder (`make_release.py`), address and version | 613 | **CLEAN** after one rule: GitHub no-reply addresses are public by design (`EMAIL_OK`), and the owner's is the public commit's author address (decision F); history 1,113 blobs, no new findings |
| 2026-10-17 | step 6, option A: shard maps (`data/corpus/shardmaps/`), the state tables asset, the published score report, `reproduce_gate.py` and its notebook | 627 | **CLEAN**; zero findings in any new file (each also inspected string by string: suffixes, counts and numbers only) |
