# V2 — the hybrid tokenizer: design

- **Status: DRAFT, 2026-10-09. Documents only.** No V2 code, data, tokenizer or training
  exists. Nothing runs before the owner approves `PREREG_V2_P1.md` (gate A).
- **What V2 is:** a new programme that changes how the engine writes the path, query and
  fragment: in learned multi-character pieces (BPE) instead of single characters. The host
  stays character by character, as `DESIGN.md` §4.2 always intended.
- **Why now:** the word-level finding (`research/e18/STRONG_DISC_RESULT.md`;
  `research/E3_FINAL_REPORT.md` §8).
- **Not changed by V2:**
  - E3 (CLOSED, NOT MET);
  - `e16_b5` (`uie/baseline.py` `BASELINE`, frozen);
  - everything under `uie/`;
  - gate v2, its thresholds and tolerances;
  - every published result and every closed branch.
- **First step:** the pilot, `research/v2/PREREG_V2_P1.md`.
- **Revised 2026-10-09, the same day, before gate A** (`PREREG_V2_P1.md` §14, Amendment 1):
  - the owner dropped the twin, set an absolute pass mark (D1-R CLEARED), added control
    mode, and moved scoring to Kaggle;
  - this supersedes, for P1, the twin in §0 and §8, §6's two matching rules (P1 now keeps
    long rows and buckets by its own length), and the character-arm checks in §7;
  - the rest of this design stands.

---

## 0. In plain words

- **The problem.** The engine writes paths and queries one letter at a time, and that is
  how it gives itself away.
  - A detector that knows which words occur in real URLs catches it: a generated benign URL
    holds 1.45 words never seen in real data, a real one 0.93.
  - Every letter looks plausible, so detectors that read letters (the gate's included) miss
    it. A word-aware detector does not: excess +0.168 benign, +0.164 phishing, CONVICTED.
- **The idea.** Give the engine bigger building blocks for those parts.
  - A standard method, byte-pair encoding (BPE), reads the training paths and queries. It
    finds the two neighbouring pieces that occur together most often and glues them into
    one new piece, and repeats that a fixed number of times.
  - Common chunks such as `index`, `wp-content/` or `utm_source=` end up as single pieces.
  - The analogy: a stamp set of whole words instead of single letters. A word spelt letter
    by letter can come out wrong at any letter; a whole-word stamp cannot be misspelt.
  - Rare and new words can still be written. Every single letter stays a piece, so nothing
    the engine can write today becomes unwritable.
- **What stays letter by letter: the host** — the name (`shopwave`) and subdomains (`cdn`).
  - That is where the engine has to invent new names, and spelling is what lets it
    (`DESIGN.md` §4.2).
  - The cost of that choice, stated plainly: V2 can only work on the path/query share of the
    word gap. The host's share (55% → 48% known words, benign) is outside it.
- **What does not change:** the plan, the slash rules, B5, the supplied suffix, the training
  objective, the data, the model's width and depth, and the gate.
- **How it is tested first:** a pilot trains two engines that differ only in this choice —
  the new one, and a letter-by-letter twin trained the same way — and compares them, with
  `e16_b5` as the fixed reference.
- **What it does not touch:** E3 stays closed and `e16_b5` stays the baseline, whatever the
  pilot finds.

## 1. The owner's request, verbatim (2026-10-09)

> Task, documents only (no code, no tokenizer changes, no training): draft
> research/v2/DESIGN_V2.md and research/v2/PREREG_V2_P1.md for a V2 hybrid tokenizer:
> character-level for the host label and subdomains, BPE for path, query and fragment, as
> DESIGN.md §4.2 intended. Cover:
>
> how the existing plan, slash and B5 rules carry over;
> the hypothesis, controls, metrics, thresholds and failure criteria;
> a small pilot comparing a hybrid model against a matched character-only model, with
> frozen e16_b5 as the reference;
> code isolation in research/v2/.
> Keep E3 closed and e16_b5 frozen. Commit, push, then stop and show me the pilot and its
> cost before running anything.

## 2. Why: the evidence

### 2.1 The word-level tell

From `research/e18/STRONG_DISC_RESULT.md`. The test was registered before it ran. It read
`e16_b5`'s free files against real held-out URLs, with a vocabulary built from 2.8M real
training URLs.

| | benign: real → `e16_b5` | phishing: real → `e16_b5` |
|---|---|---|
| words never seen in training, per URL | 0.93 → **1.45** | 0.76 → **1.06** |
| longest never-seen word (characters) | 8.9 → 10.3 | 6.0 → 7.4 |
| known words in the **path and query** | 94% → **84%** | 96% → **93%** |
| known words in the **host** | 55% → 48% | 78% → 71% |
| word-aware detector (D1), whole URL | excess **+0.1675, CONVICTED** | excess **+0.1637, CONVICTED** |
| the same detector, structural features only | AUC 0.502 | AUC 0.552 |

- The whole tell is in the words. Lengths, counts and depth carry none of it.
- A "word" there is a run of three or more letters (`[a-z]{3,}`, lower-cased), looked up in
  the class's real training URLs.

### 2.2 The findings it ties together

- **The engine is too inventive** (`E3_FINAL_REPORT.md` §5.3 b). In E12, 77.2% of its
  phishing queries were never seen in training, against a real 50.6%; 16.7% were exact
  training queries, against a real 40.3%.
- **A simple model does better on the same data** (§5.3 d). A character 6-gram model
  trained on the same 8,272 phishing training queries scores +0.036 where the transformer
  scores +0.124. The n-gram model reproduces local real strings; the transformer
  generalises past them.
- **The query tell is lexical** (`research/qc1/QC1_RESULTS.md` §4). Flattening letters and
  digits removes about half of benign's query tell and four-fifths of phishing's.

### 2.3 The mechanism, as a hypothesis [HYPOTHESIS]

A letter-by-letter writer can produce words that do not exist in two ways:

1. **Slips.** Every letter is a chance to leave a real word, and small chances compound. If
   each letter of an 8-letter word stays on a real word with probability 0.97, the word
   comes out real 0.97⁸ ≈ 78% of the time.
2. **Fluent invention.** The engine learns how real words are *built* and composes new
   ones that follow those rules: `priroditeley`, `khatiyunktiyurut` (E18's example).

Pieces work against both:
- a common word becomes one decision, which removes the slips;
- composing a new word takes several piece choices that real paths rarely make, which
  should make invention rarer.

Neither cause has been measured directly. QC1-4 found no excess of near misses at the level
of whole queries (NO BLENDING), which argues that slips are not the main cause there. **The
pilot tests the consequence — fewer invented words in the fields that get pieces — not the
cause.**

## 3. Prior arguments, stated

The programme's rule: before a lever goes into a plan, the record's arguments against it are
found and answered.
- **Searched:** `ARCHITECTURE_DECISION.md`, `E3_REPORT.md`, `research/qry/REPAIR_PLAN.md`,
  `research/control/REPAIR_PLAN.md`, `NATIVE_REPAIR_PLAN.md`, the decision records, QC1,
  E13, and the E12–E16 roadmaps and designs.
- **Found:** every record in the table below.

| record | what it says | what it measured | bearing on V2 |
|---|---|---|---|
| `research/qc1/QC1_RESULTS.md` §3 | query subwords **not licensed** for E12 | whole query strings: the mix of copied and new strings (no MIX-LED); new strings one edit from a frequent training query (no BLENDING); parameter names (not IDENTIFIED) | A licence rule for choosing E12's levers, not a finding that pieces cannot help. None of the three instruments looks at the words inside a string. |
| `research/qc1/QC1_RESULTS.md` §4 | "It is the strings, not the rate": the engine's copied queries are themselves detectable (copied against copied 0.15–0.17), so copying *more* is not the fix | which strings are copied | **A real risk for V2.** Pieces make common strings cheap to write, so the engine could copy the most common strings even more often — the wrong way. Guarded: copying is checked two-sided against the natural rate (G7), and D1 sees word frequencies. |
| `research/e13/E13_RESULTS.md` Appendix B (E13-P) | path and subdomain subwords "**ruled out by these instruments**" | QC1's mix and near-miss rules on whole paths and subdomains; QC1's names route was not run | As QC1: whole-string instruments, and the record itself limits the ruling to them. |
| `research/native/NATIVE_REPAIR_PLAN.md` N2c | tokenization (N2c-T) is a branch taken "only on N2a's reading" | — | N2a read SPARSITY (benign; phishing UNAVAILABLE), so N2c was never licensed, and within E3's rules tokenization stayed closed. V2 is a new programme, opened by the owner on evidence that came later (E18). |
| `E3_REPORT.md` §13.4 and §20 (E5) | "PTH tell is segment vocabulary": **refuted**. Generated path segments were 56.4% in the training vocabulary, against a 48.5% real floor | whole path segments (between slashes), on E5 | Not a contradiction of E18. It was a different engine (before E9's objective fix, when paths were short and common) and a different unit (segments, not words). It does show that a vocabulary reading depends on the unit and the engine, so the pilot reports both units. |
| `research/native/DECISION_2026-09-29.md` §4 (R3) | capacity is the last resort | — | V2 adds no width or depth. It adds one embedding row per piece: 8,000 × 512 = 4,096,000 parameters (+16%), usable only where pieces occur. R3's own order puts representational candidates before capacity. The pilot also reads whether E16's 1.77× capacity step moved the word tell (`e15_new` against `e16_b5`, from existing files). |
| `E3_FINAL_REPORT.md` §5.3 c | fit has decoupled from the tell | — | V2 is judged on tells, never on fit. Fit is a guard only. |
| `E3_FINAL_REPORT.md` §5.3 a | 63% of the free phishing QRY tell comes from the host and path written before the query | — | V2 is not expected to clear that cell, and the pilot does not target it. |
| `research/native/E12_ROADMAP.md` §2.4 and §6 | "Subwords are the most promising lever, and the most expensive"; the risk that "the model can over-reproduce frequent tokens" | — | Carried into the guards. The same section records that `DESIGN.md` §4.2 specified subwords, that E2 built a character tokenizer (commit 079592f), and that "no later document argues against subwords". |

**What is new since every ruling above:** E18 measured the tell at the level of words, which
no earlier instrument was built to see, and found it in both classes.

## 4. The V2 tokenizer

### 4.1 The sequence: unchanged except inside three fields

```
e16_b5:  <bos> <C:benign> <F:domain> <SUF> c o m <P:1> <Q:1> <S:1>
         <LBL> s h o p w a v e <SUB> c d n
         <PTH> s t a t i c / v 2 / b u n d l e . j s <QRY> i d = 4 2 <eos>

V2:      <bos> <C:benign> <F:domain> <SUF> c o m <P:1> <Q:1> <S:1>
         <LBL> s h o p w a v e <SUB> c d n
         <PTH> [static/] [v2/] [bundle] [.js] <QRY> [id=] [42] <eos>
```

- The bracketed pieces are illustrative; the real ones are learned.
- Every marker, control token, plan token and `<RS>` is the same token, in the same order.
- Only the contents of `<PTH>`, `<QRY>` and `<FRG>` change.

### 4.2 Which field is written how

| field | `e16_b5` | V2 | why |
|---|---|---|---|
| suffix (`SUF`) | characters, supplied | characters, supplied | `DESIGN.md` §4.3: a new TLD works from its spelling |
| registrable label (`LBL`) | characters | **characters** | where invention happens (§4.2) |
| subdomains (`SUB`) | characters | **characters** | where spoofing structure lives (§4.2) |
| userinfo (`USR`), port (`PRT`) | characters | characters | host-side and rare; §4.2 is silent, so nothing changes |
| path body (`PTH`) | characters | **pieces (BPE)** | §4.2 |
| query (`QRY`) | characters | **pieces (BPE)** | §4.2 |
| fragment (`FRG`) | characters | **pieces (BPE)** | §4.2 |

### 4.3 The pieces

- **The base alphabet is the character vocabulary** (`e16_cap`'s, `min_count` 1). Every
  single character stays a piece, so the hybrid can write every string the character engine
  can.
- **Learned once, from training data only.**
  - The data: the path, query and fragment bodies of the training rows the character engine
    keeps, plus the query bodies of E15's stream.
  - Each body is weighted by how often training draws its row (class α = 1, exactly as the
    sampler draws), so the pieces reflect the data as the engine sees it.
  - Never validation rows, never the gate's pools.
- **One table for the three fields.** Each field body is its own sequence, so a piece never
  spans two fields. The `?` and `#` that separate fields stay with the grammar, as now.
- **No pre-splitting inside a field.** Pieces may cross `/`, `&`, `=` and `-`. `DESIGN.md`
  §4.2 is explicit: "a valid pattern becomes one atomic unit instead of three independent
  chances to go wrong".
- **The method: standard greedy BPE.**
  - At each step, the adjacent pair that occurs most often becomes one piece.
  - Ties go to the pair whose two pieces come first in code-point order.
  - **8,000 merges**, fixed now and never tuned on an outcome.
- **Encoding is canonical.** Merges are applied in the order they were learned, so each
  string has exactly one encoding.
- **Ids.** Every existing token keeps its id, and the 8,000 pieces are appended after them.
  That lets both pilot engines start from the same weights (`PREREG_V2_P1.md` §4.2).

### 4.4 The piece rule (decode)

A piece may be sampled only inside a field whose training encodings use it. Single
characters stay legal wherever they are legal today.

- **Host fields are therefore written in characters only, in both engines, by
  construction.**
- **It forbids no string.**
  - Everything `e16_b5` can write, the hybrid can write. A piece only adds a second way to
    write a string that occurs in real fields.
  - That makes it weaker than D1 and B5, which forbid events. This rule removes token paths,
    never URLs.
  - It has no fitted parameter.
- **The mass it removes is recorded per field.** A large removal would itself be a finding:
  the engine confusing which field it is in.

### 4.5 Lossless by requirement

For every training, validation and stream URL, the hybrid must give back exactly what the
character tokenizer gives back:

    hybrid.decode(hybrid.encode(u)) == char.decode(char.encode(u))

Zero failures are allowed. This is the standard the factorization met on 2,352,720 URLs.

### 4.6 Likelihood with pieces

- A string can be split into pieces in more than one way, and the engine is trained on the
  canonical split only. Its probability for that split slightly underestimates the string's
  probability.
- **Every comparison of fit between the two engines is therefore in nats per character of
  the URL**, computed from the canonical split. That is fair to the character engine and
  conservative for the hybrid.
- How often the hybrid samples a non-canonical split is reported.

## 5. How the plan, slash and B5 rules carry over

The three rules act only on structural tokens. None of them looks inside a field.

| rule | what it acts on | in V2 | why it is unchanged | checked by |
|---|---|---|---|---|
| **presence plan** `<P:v> <Q:v> <S:v>` (E8, E11, E12) | three tokens right after the suffix | same tokens, same place, same supply tables (the presence-slash table `5a0eeac9…` and the shipped suffix table) | the plan is written before any field body, so no piece can reach it | following, read from the URL strings (G6′) |
| **root slash** `<RS>` (E8) | one marker for a bare `host/` | unchanged | the bare root slash is a marker, never a piece; a path body is non-empty by the existing rule | the slash rows (G5′) |
| **D1 / D1′**, the slash rule (E13, E14) | the steps that end the host, over its five end tokens: `<PTH>` `<RS>` `<QRY>` `<FRG>` `<EOS>` | unchanged | the host is still written in characters, the five end tokens are the same tokens, and D1 stops when the host ends, before any piece can be sampled | S honoured exactly on every plan the rule covers (V-rules) |
| **B5**, the query-plan rule (E17) | every step from the label marker to `<QRY>`: `<EOS>` and `<FRG>` get probability 0, and their mass goes to `<QRY>` | unchanged | the forbidden events are the same: ending, or starting a fragment, before the query. Path steps are now piece steps. There are fewer of them, but each is constrained exactly as before | no URL under a Q = 1 plan ends or starts a fragment before its query (K17-B2, as amended; V-rules) |
| **the supplied suffix** | characters injected after `<SUF>` | unchanged | `SUF` stays characters | — |

**One difference in detail, stated.** B5 moves an ending's mass to `<QRY>` "where the URL
would have ended". With pieces, the path can end only at a piece boundary, so that point is
a piece boundary rather than any character. The rule's guarantee is unchanged: no query plan
without its query. Its moved mass is recorded per step, as now.

## 6. How the objective carries over

- **E9's fixed-constant objective is maximum likelihood over whole URLs, whatever the
  unit.** A URL's loss is the sum of −log p over its tokens, which is −log p(URL). Pieces
  change how the sum is split, not what it adds up to.
- **The constant C is re-derived by E9's own rule** (`PREREG_E9.md` D3: batch size × the
  expected scored-token mass per drawn row, old rows only, as for `e12_final`'s pinned
  value).
  - Under AdamW, a constant scale on every gradient has no material effect on the updates.
  - Re-deriving it keeps the logged gradient scales comparable between the two engines.
  - The value is recorded before training.
- **Field masks, the frame token and the stream's query-only targets are unchanged.** A
  piece inside `QRY` is a query target, as a character was.
- **Clipping:** the same schedule (1.0, then 5.0 after step 1,000). K12e (at most 1% of steps
  clipped per band after step 1,000) applies to both engines.

**Two matching rules that a straightforward implementation would break:**

1. **Dropped rows.** Training drops any row longer than `max_len` (384 tokens). Pieces
   shorten rows, so the hybrid would keep long rows the character engine never trains on: a
   data difference posing as a tokenizer difference. **Rule:** both engines train on the
   character engine's rows. A row is kept if and only if its character encoding fits 384.
   The same holds for validation rows.
2. **Batches.** The sampler sorts each window by token length before cutting it into
   batches, so different units would give different batches. **Rule:** both engines sort by
   character length, so both draw the same rows in the same batches. This is checked by the
   batch digest over the character encodings (`adbddc2a…`, `e16_cap`'s).

## 7. Code isolation

- **Nothing under `uie/` changes in V2-P1.**
  - `BASELINE` stays `E16_B5`.
  - Tests pin `uie/`'s tree at the commit V2's code starts from, and `E16_B5`'s weight and
    file hashes.
- **All V2 code lives in `research/v2/`.**
  - Tests follow the repo's convention: `tests/test_v2_*.py`.
  - Data goes to `data/v2/` (gitignored, like all data).
  - The Kaggle bundle is `uie-v2p1-code.zip`.
- **How `uie/` is reused without being changed:**
  - `HybridTokenizer` subclasses `URLTokenizer`. It overrides three things:
    - the encoding of the three fields;
    - `decode_fields`, which recognises pieces by id range (today's decoder keeps only
      one-character tokens and would silently drop them);
    - the state dict.
  - Generation calls `uie.model.generate` as it is. On the decode path `e16_b5` uses, the
    only one-character assumption is the tokenizer's own decode. The piece rule is applied by
    a thin model wrapper in `research/v2/`.
  - Training reuses `uie.train`'s parts (the field mask, the stream injector, accumulation,
    the fixed objective) with a V2 sampler keyed on character length.
- **Equivalence, proven before anything runs on Kaggle:**
  - V2's trainer in character mode reproduces `uie.train.train`'s first steps bit for bit on
    the CPU, and its batch digest over 20,000 steps equals `adbddc2a…`;
  - V2's free-mode generator, run with `e16_cap`'s weights and `e16_b5`'s decode on the CPU,
    reproduces the start of `e16_b5`'s published free files byte for byte.
- **Planned modules,** written only after gate A:

| file | role |
|---|---|
| `v2_common.py` | constants, hashes, seeds and recipes; the frozen references |
| `v2_bpe.py` | the BPE learner and encoder (pure Python, deterministic) |
| `v2_tokenizer.py` | `HybridTokenizer` |
| `v2_start.py` | the shared start: `e16_cap`'s start plus the piece rows |
| `v2_train.py` | training for both arms, with the matching rules of §6 |
| `v2_generate.py` | free-mode generation, with each URL's plan recorded, and the piece rule |
| `v2_verify.py` | the validity checks |
| `v2_words.py` | the word-level readings (E18's detector, imported unchanged) |
| `v2_score.py`, `v2_compare.py` | the gate's free cells, the guards and the criteria |
| `make_v2_bundle.py`, `make_v2_notebooks.py`, `V2P1_*.ipynb`, `V2P1_KAGGLE.md` | the Kaggle side |

- **No V2 engine becomes a baseline inside V2-P1.** Adoption is a separate owner decision,
  with its own binding step.

## 8. The V2 programme

| phase | what | status |
|---|---|---|
| **P1** | the pilot: the hybrid against a matched character twin, free mode, `e16_b5` as the reference | registered in `PREREG_V2_P1.md`; awaiting gate A |
| P2 | the full reading of a V2 engine: control mode, all 32 cells, all 8 slash rows | only if the owner opens it after P1 |
| later | anything else, for example the host's share of the word gap | the owner's decision |

- **The word detector becomes a standing V2 instrument** from P1 on, as E18 §5 recommended,
  so any improvement is measured rather than assumed. The gate itself is unchanged.
- **A V2 finish line is the owner's decision.** A proposal only, not in force:
  - E3's two conditions met by a V2 engine;
  - plus the word detector on the path, query and fragment (D1-R, defined in the pilot)
    CLEARED in both classes.
  - The host's share of the word gap is not in it, because V2 leaves the host as it is.

## 9. Risks

- **Over-copying.** Pieces make the most common strings cheap: the risk `E12_ROADMAP.md` and
  QC1 §4 point to. Guarded two-sided against the natural rate, for queries (G7, existing)
  and for paths (G7-P, new).
- **New tells from pieces.** The engine could join real words in combinations real URLs
  never use.
  - The char-level gate (G1′, G2′) and the whole-URL detectors guard against this, and D1's
    frequency features see unusual word mixes.
  - A coherence tell that no current instrument reads would go unseen. The report states
    that limit.
- **Phishing power.** Close to 60% of free phishing URLs have no path, query or fragment, so
  the phishing reading rests on fewer URLs and may be void.
- **Fit without tell.** The programme saw fit improve while tells stood still (E16, E17). The
  pilot is read on tells.
- **The host gap remains.** V2 cannot clear the whole-URL word detector while the host is
  written letter by letter. That is a design choice, stated here so that it is not mistaken
  later for a failure.

## 10. Records

| item | location |
|---|---|
| the original design | `DESIGN.md` §4.2 |
| the word-level finding | `research/e18/STRONG_DISC_RESULT.md`, `STRONG_DISC_PREREG.md`, `data/e18/strong_disc/` |
| E3, closed | `research/E3_FINAL_REPORT.md` (§8 is the addendum) |
| the frozen reference | `uie/baseline.py` `E16_B5`; `research/native/DECISION_2026-10-08.md`; `data/e17/decode/gen` |
| the pilot | `research/v2/PREREG_V2_P1.md` |
