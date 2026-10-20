# V2 — a review of the plan, and a revised plan

- **Status: a review for the owner's approval, 2026-10-09. Not a registration.** It changes
  nothing by itself: no code, no tokenizer change, no training, no run.
- **Requested** after R0 and before implementation. The request is quoted below.
- **Unchanged:** E3 stays CLOSED. `e16_b5` stays `BASELINE` and frozen. `uie/` is untouched.
- **The current registration is paused** (`PREREG_V2_P1.md`, with Amendment 1 and the gate A
  record).
  - If this review is approved, R1 gets its own registration first.
  - P1's registration is then rewritten after R1 reads.
  - The old text stays in the record.

## The owner's request, verbatim

> I want a scientific and practical review of the current V2 plan before approving
> implementation.
>
> 1. Is BPE actually the best approach? Explain the scientific reasoning and evidence for
> choosing it over character-level modeling or other suitable alternatives. Do not justify it
> merely because BPE appeared in the original design. Compare the options and identify their
> trade-offs.
>
> 2. Address the host problem directly. E18/R0 found a substantial word-level realism tell in
> hosts, especially for phishing-style URLs. Why does the current plan leave hosts
> character-level while applying BPE only to paths, queries, and fragments? Evaluate whether
> the host needs a different representation, training approach, or another solution. Do not
> exclude it simply because the current plan does.
>
> 3. Start from the actual objective: UIE must generate creative, novel URLs while correctly
> representing real-world URL structure and naming patterns—not merely pass a detector.
> Review whether the proposed changes can achieve that objective without increasing copying
> or damaging validity, novelty, diversity, or controllability.
>
> 4. Propose the best research plan. Reassess the current plan, compare credible
> alternatives, and recommend a staged, preregistered experiment with measurable hypotheses,
> controls, evaluation metrics, failure criteria, and realistic compute costs. Separate
> evidence from assumptions and explain what each experiment would teach us.
>
> Keep E3 closed and e16_b5 frozen. Do not write code, change the tokenizer, modify the
> baseline, or run experiments yet. First return the review and revised plan for my
> approval.

---

## 0. In plain words

**The four verdicts**

1. **BPE is a sound first test, not a proven best choice.**
   - The problem we measured is that the engine invents words far more often than real URLs
     do. That is the classic failure of letter-by-letter generators, and word pieces are the
     standard and cheapest remedy.
   - But whether pieces work here depends on something nobody has measured yet. If the
     engine's fake words stand in for *common* real words, pieces can capture them. If they
     stand in for *rare* ones, the engine would still assemble fakes, only from pieces
     instead of letters.
   - That can be measured on the laptop, from files we already have, before any GPU time.
2. **The host should not be left out.**
   - `DESIGN.md` kept hosts letter by letter so the engine could invent names. But pieces
     can invent names too, real names are mostly built from real words and fragments, and
     the problem measured in hosts is the same over-invention.
   - For phishing, the host is the *larger* half of the word problem: +0.147, against +0.100
     for the path, query and fragment.
   - Hosts also carry extra risks: copying real domains, regenerating live phishing domains,
     falling into cliché names. So they get their own arm and their own guards.
3. **One detector is too narrow a pass mark for the real goal.**
   - An engine can satisfy a word-list detector by copying real URLs, or by stringing real
     words together in orders real URLs never use.
   - The revised plan judges six things: realism of letters, of words and of word order;
     new words at the real rate; no extra copying; diversity; validity; control.
4. **The revised plan has three stages.**
   - **R1:** a diagnosis on the laptop, with no training.
   - **P1:** two engines trained side by side and compared with frozen `e16_b5`: pieces in
     the path, query and fragment (arm A), and pieces in the hosts as well (arm B).
   - **P2:** either a second training of a success, to confirm it, or the next lever the
     evidence points to.

**The cost**
- **R1:** about 3–5 hours of laptop computing; no Kaggle.
- **P1:** about 10–11 Kaggle GPU-hours, 4–6 hours of Kaggle CPU sessions and about 10 laptop
  hours; 5–7 days.

**What I got wrong before.** My design kept hosts as letters because `DESIGN.md` §4.2 did. I
never tested that reason against the host reading. Your question is right to press on it.

---

## 1. What we know, and what we only assume

### 1.1 Measured in this project

| finding | the numbers | source |
|---|---|---|
| A word-aware detector catches `e16_b5`'s whole URLs | excess +0.1675 benign, +0.1637 phishing, CONVICTED | E18 |
| Structural features alone do not | AUC 0.50 benign, 0.55 phishing | E18 |
| The word tell is in both parts of the URL | path, query and fragment: +0.192 benign, +0.100 phishing; host: +0.068 benign, +0.147 phishing | R0 |
| Fewer real words in the path, query and fragment | known words 77.3% against a real 91.9% (benign), 78.4% against 83.1% (phishing); never-seen words per URL 0.79 against 0.28, and 0.50 against 0.41 | R0 |
| Fewer real words in the host | known words 48% against a real 55% (benign), 71% against 78% (phishing) | E18 |
| Width helps only a little | width 384 → 512, +77% parameters: path/query/fragment −0.023 benign, −0.004 phishing (not significant); host −0.011 benign, −0.022 phishing | R0 |
| The engine copies whole queries *less* often than real data does | free benign 23.5% against a natural 31.5%; free phishing 24.1% against 41.9% | E17 scoring (`copy.log`) |
| Real phishing reuses its strings; the engine invents | phishing queries that are exact training queries: 40.3% real, 16.7% the engine (E12); never seen: 50.6% real, 77.2% the engine | `E3_FINAL_REPORT.md` §5.3 b |
| A plain letter 6-gram model is closer to real on phishing queries | +0.036, against the transformer's +0.124, on the same data | §5.3 d |
| Better fit has not moved the tells | E16: fit −0.10 nats per character, far cell −0.0018. E17: fit better, cell worse | §5.3 c |
| The query tell is about content, not shape | flattening letters and digits removes about half of benign's tell and four-fifths of phishing's; new queries are not near misses of frequent ones | QC1 §4, QC1-4 |
| On an earlier engine (E5), path segments were not the tell | 56.4% of generated segments seen in training, against a 48.5% real floor | `E3_REPORT.md` §13.4 |
| Two trainings of one recipe differ | by up to 0.0148 on a gate cell | E12 |

### 1.2 From the research literature (outside this project; not measured here)

- **Letter-level generators characteristically produce pronounceable words that do not
  exist.** Word-piece generators (BPE: Sennrich et al., 2016; unigram pieces: Kudo, 2018)
  produce far fewer, and became the standard for open-vocabulary text because they handle
  rare words with a modest vocabulary.
- **Hybrid word-and-letter models** (for example Luong & Manning, 2016) keep common words as
  units and spell rare ones letter by letter.
- **Byte and letter models can match word-piece models,** but need more depth and compute
  per string (for example ByT5, Xue et al., 2022).
- **Memory-based language models** (kNN-LM, Khandelwal et al., 2020) raise the reuse of
  real strings by looking them up at generation time.
- **Pretrained language models carry a strong prior over real words,** and fine-tuning
  passes it on.
- **Training a text generator against a discriminator** (adversarial or reinforcement
  training) tends to collapse its diversity, and rarely beats plain likelihood training
  (for example Caccia et al., 2020). `DESIGN.md` §4.6 says the same.

### 1.3 Assumptions (untested here), and the stage that tests each

| | assumption | tested by |
|---|---|---|
| A1 | Pieces reduce the engine's fake-word rate in URLs | P1 |
| A2 | The fake words stand in for *common* real words, which pieces can capture, rather than rare ones | R1 |
| A3 | Pieces in hosts reduce the host tell without raising copies of real domains or of live phishing domains | P1, arm B |
| A4 | A word-list detector is a fair judge of naming realism. Partly checked (E18's positive controls). It can be passed by copying, or by real words in unreal orders | R1 (a word-order detector) and P1's copying guards |
| A5 | One training stands for its recipe. Run-to-run spread on the word readings is unknown; 0.023 between two letter engines is the only yardstick | P2 (a second training before any adoption) |
| A6 | Kaggle's software scores like the laptop's | V-env: `e16_b5` scored again in the same run |

---

## 2. Question 1: is BPE the best approach?

### 2.1 The problem, stated as something we can measure

- **Real URLs do contain words never seen before:** 0.28 per benign path, query or fragment,
  and 0.41 in phishing. The engine writes 0.79 and 0.50.
- **So the target is not "no new words".** It is "new words at the real rate".
  - Too many new words look fake.
  - Too few means copying.
- **Every option below is judged against that two-sided target,** not against a detector
  alone.

### 2.2 Why a letter-by-letter engine over-invents [INTERPRETATION]

Three mechanisms, which do not exclude each other:

1. **Slips.** Every letter is a chance to leave a real word, and small chances compound
   over a long word.
2. **Fluent invention.** The engine learns how words are *built* and composes new ones that
   follow the rules (`priroditeley`, E18's example).
3. **Too small a memory for the vocabulary.** The engine has to hold millions of real words
   as letter patterns in its weights. It keeps the common patterns and smooths over the
   rest, producing words that fit the patterns but do not exist.
   - Widening it helped a little (−0.023), which fits this account, but far too little to
     be the remedy.
   - Fit kept improving while the tells stood still (E16, E17): the extra capacity went
     elsewhere.

**Which mechanism dominates is not measured.** R1-2 and R1-3 measure it, and the answer
decides which remedy fits.

### 2.3 The options

| option | what it changes | why it could work | evidence | risks and costs | verdict |
|---|---|---|---|---|---|
| **More width** (R3) | nothing in the representation | a bigger memory for the vocabulary (mechanism 3) | **measured:** +77% parameters bought −0.023 of a 0.156 gap (benign), nothing in phishing | many times more width for a fraction of the gap; locked as the last resort | not a route |
| **More steps** (R2) | longer training | more exposure to rare words | E16's plan parts overfitted after 7k steps | memorisation; rejected | not a route |
| **Cooler sampling** (temperature, top-p) | the decoder only | cuts the low-probability tail, where fake words live | none here | closed by you. It also cuts novelty everywhere and pushes toward cliché (`DESIGN.md` §4.6), and choosing a setting by the detector is tuning to the metric | not a route |
| **Word pieces: BPE** (the current plan) | path, query and fragment written in learned pieces | common words and fragments become single units: fewer decisions per word, and a cheap memory of the head of the vocabulary | **literature** (§1.2); none here yet | rare words are still assembled from pieces, so fluent fakes can survive in the tail. It can over-copy common strings (QC1 §4). +16% parameters at 8,000 merges | **the right first test, if R1 finds the shortfall in words that pieces can capture** |
| **Unigram pieces** (SentencePiece) | as BPE, a different way of choosing the pieces | the same | **literature:** usually close to BPE | the same | no reason to prefer it for a first test |
| **Lexicon + spelling** (common words as units; rare or new words spelled in letters) | the path, query and fragment, and possibly the host | makes "a new word here" one explicit decision, learned from how often rare words occur in training: new words at the real rate *by construction* | **literature:** works, though word pieces generally won in translation | rare *real* words would still be spelled in letters, so fakes can survive. A large lexicon costs parameters. Defining a "word" in a URL needs rules (digits, hyphens). Units that match the detector's own definition of a word risk teaching to the test | the strongest rival. Second in line, or first if R1 finds the shortfall in mid-frequency words that a lexicon covers and pieces do not |
| **A deeper letter model** (ByT5-style), from scratch | nothing in the representation | depth stands in for units | **literature:** needs more compute | the same smoothing problem, at a higher cost | not first |
| **Memory and copying** (kNN-LM, a pointer into a lexicon) | adds a lookup of real strings at generation | real words come from memory | **literature** | raises copying by design; slower generation; complex | later, only if units fall short |
| **A pretrained language model as a prior** (fine-tune a public model on the corpus) | the whole model | the strongest known source of real-word knowledge | **literature:** strong | an external model and its licence; it may have memorised real URLs, including phishing ones (copying, safety); much more compute; it changes the project's "learn from the web corpus" footing; needs your approval | the most powerful later option, not the first |
| **Training against a word detector** (adversarial, reinforcement, penalising unseen words) | the objective | pushes away from fake words | **literature:** collapses diversity | tuning to the metric; mode-seeking (`DESIGN.md` §4.6) | rejected |
| **Filtering out URLs with unseen words** | a post-hoc rule | removes fake words | — | a rule acting as the generator; tuning to the metric; it also removes the real rate of new words | rejected |

### 2.4 Verdict

- **BPE is the best first test, for three reasons that do not depend on `DESIGN.md`:**
  1. The measured symptom, over-invention of words, is exactly what word pieces are known to
     reduce in letter-level generators.
  2. It is the cheapest change that keeps every rule, the plan, the objective and the
     pipeline as they are.
  3. It fails informatively: per-field readings show how much it moved, and where.
- **It rests on an untested assumption (A2).** If the fake words mostly stand in for rare
  real words, BPE's pieces will still assemble fakes. The lexicon route, or later the
  pretrained prior, would then be the better bet.
- **R1 measures A2 before any GPU time is spent.**

---

## 3. Question 2: the host

### 3.1 The evidence

| | benign | phishing |
|---|---|---|
| host word tell (D1-H, `e16_b5`) | +0.068 | **+0.147** |
| path, query and fragment word tell (D1-R, `e16_b5`) | **+0.192** | +0.100 |
| known words in the host, real → `e16_b5` | 55% → 48% | 78% → 71% |
| effect of width 384 → 512 on the host | −0.011 (t 2.3) | −0.022 (t 7.7) |

- **For phishing, the host is the larger half of the word problem.** A V2 that leaves hosts
  as letters could at best fix the smaller half for phishing, and could never make phishing
  URLs pass a whole-URL word detector.
- **Real phishing hosts are built from a small stock of words:** brand names, `secure`,
  `login`, `verify`, `support`, `account`. 78% of their words are known, against 55% for
  benign hosts. They are as repetitive as phishing queries. The engine invents letter
  patterns instead.

### 3.2 The reason hosts were left as letters, tested

`DESIGN.md` §4.2 says the label is "where invention happens: coinages, mimicry,
hyphenation", and "must handle unseen strings". Three things cut against it:

1. **Pieces can write unseen strings.** Single letters remain pieces, so nothing becomes
   unwritable.
2. **Real names are mostly built from real words and fragments:** `northvale-clinic`,
   `shopwave`, `paypal-help`. Invention in real data happens at the level of *combinations*,
   which pieces represent naturally (§4.6's "combinations that never co-occurred in
   training").
3. **The measured host problem is the same over-invention** as in paths.

So excluding hosts was a design assumption, not a finding.

**What letters do that pieces might not.** One-letter edits of real names (typosquats such as
`paypa1` or `rnicrosoft`) are a real phishing pattern. With pieces they stay writable,
because letters remain. Whether the engine still writes them at the real rate is measured:
R1-3 sets the real rate, and P1 checks it.

### 3.3 What could go wrong with pieces in hosts

| risk | why pieces raise it | how it is measured |
|---|---|---|
| copying real domains | whole known names become cheap to write | exact copies of training registrable domains (`uie/gate.py` NOVELTY), against `e16_b5` |
| regenerating live phishing domains | the same, in the phishing class | collisions with the offline threat snapshot (`uie/gate.py` SAFETY), against `e16_b5` |
| cliché names | the most common combinations become the easiest (`secure-login-verify`, `DESIGN.md` §4.6) | duplicates within a run; distinct words per 1,000 at equal n, against real |
| losing typosquats | whole-word pieces may crowd out one-letter edits | the near-miss rate (one edit from a common word), against real |

- **The suffix stays in letters.** It is supplied from a table, and its spelling is what
  lets a new TLD work without retraining (`DESIGN.md` §4.3).
- **D1 and D1′ are unaffected.** They act on the five host-ending markers, which do not
  change.

### 3.4 The options for the host

| option | what it means | verdict |
|---|---|---|
| H0, keep letters | accept the host tell | rules out word-level realism for phishing for good |
| **H1, pieces in labels and subdomains** | a host table of pieces, separate from the path table; suffix in letters | **test it as its own arm (B)** |
| H2, lexicon + spelling in hosts | as §2.3 | second in line, as for paths |
| H3, a pretrained prior | as §2.3 | later, only if units fall short |

### 3.5 Recommendation

- **Test pieces in hosts in P1, as a separate arm (B).** Then A against B isolates the host
  change: same data, same batches, same start.
- **Give B host-specific guards:** domain copies, threat-snapshot collisions, cliché and
  diversity, and the typosquat rate.

---

## 4. Question 3: does the plan serve the real objective?

### 4.1 The objective, made measurable

UIE must invent URLs that look real in every way a careful reader could check, at the real
rate of invention, without copying and without losing control. Six properties:

| | property | in plain words | measures | target |
|---|---|---|---|---|
| 1 | realism of letters | each field looks right letter by letter | the gate: 32 cells; FULL | CLEARED, as in E3 |
| 2 | realism of words | real words at the real rate | D1, D1-R, D1-H, D1 per field; known-word share | CLEARED; within T of real |
| 3 | realism of word order | real words in orders real URLs use | **D4 (new):** a detector of word pairs and triples | no worse than `e16_b5`; CLEARED is the aim |
| 4 | new at the real rate, with no extra copying | invents new words, strings and URLs as often as real data does — no more, no less | never-seen words (two-sided); exact copies of each field against the natural rate (two-sided); copies of strings seen once (K17-3's band rule); near misses; whole-URL novelty; copies of real registrable domains; threat-snapshot collisions | within T of real, or not above `e16_b5` |
| 5 | diversity | no collapse onto favourites | duplicates within a run; distinct words per 1,000 at equal n; templates per 1,000 (E17's coverage tool) | not below real − T |
| 6 | validity and control | parses, real suffix, follows the plan | `uie/gate.py` pass rates by reason; presence; the 8 slash rows; plan following; rule integrity | as now |

### 4.2 Where the current plan falls short

- **Its pass mark is a single detector (D1-R).** A model could meet it by copying, or by
  stringing real words in unreal orders. Nothing in the plan would notice the second.
- **No judge of word order.**
- **Novelty is not targeted two-sided.** Fewer new words than real data would be copying,
  but nothing flags it.
- **Copying guards cover queries and paths only.** Nothing covers copies of real domains or
  collisions with live phishing domains, which matter most once hosts change.
- **No diversity measure,** and the validity gate's pass rates are not read.
- **Hosts are excluded.**

### 4.3 What pieces do for creativity

- **They move invention from letters to combinations.** That is the kind of creativity
  `DESIGN.md` §4.6 asks for: "combinations that never co-occurred in training". Fake words
  are a tell, not creativity.
- **The risk runs the other way: too little invention.**
  - Pieces make common strings cheap, so the engine could copy them too often or fall into
    cliché.
  - Property 4 (two-sided) and property 5 catch that, and they are guards, not reported
    extras.
- **Control is unaffected.** The plan, slash and query rules act on markers that do not
  change, and the suffix is still supplied in letters.
- **Validity:** pieces come from real strings, and single letters remain, so validity
  should not move. The gate's pass rates confirm it.

---

## 5. Question 4: the revised plan

### 5.1 The stages

| stage | question | where | cost | what it decides |
|---|---|---|---|---|
| **R1** | Where is the word tell, what kind of words does the engine get wrong, and could pieces capture them? | laptop, existing files, no training | 3–5 h of computing | P1's arms, fields and piece counts, or no P1 |
| **P1** | Do pieces fix the word tell in the path, query and fragment (A), and in hosts too (B), at no cost to the other five properties? | Kaggle (GPU train and generate; CPU scoring), laptop checks | about 10–11 GPU-h, 4–6 h CPU, about 10 laptop h | success, continue, or stop |
| **P2** | Confirm a success, or take the next lever the evidence names | depends on P1 | similar to P1 | adoption (your decision), or the next step |

### 5.2 R1 — the lexical diagnosis (laptop; registered before it runs)

**On `e16_b5`'s and `e15_new`'s existing free files, against the gate's real pools.**

| code | question | measure | what it teaches |
|---|---|---|---|
| R1-1 | Which fields carry the word tell? | E18's D1, one field at a time: label, subdomains, path, query | which fields need units; whether labels and subdomains differ |
| R1-2 | Common or rare words? | the words each source uses, by how often they occur in training (bands 0, 1–9, 10–99, 100–999, 1,000+), real against generated, per field | where the shortfall sits; tests A2 |
| R1-3 | What are the fakes like? | each never-seen word classed as a near miss (one edit from a word seen 10+ times), a join of two real words, or other. Real held-out never-seen words are classed the same way, as the reference | slips or fluent invention; the real typosquat rate in hosts |
| R1-4 | Could pieces capture the right words? | BPE learned on training data at 2k, 8k and 32k merges, separately for paths and hosts: the share of real held-out words that come out as a single piece, per field | the most pieces could fix; whether 8k is enough |
| R1-5 | Is word order already a tell? | **D4:** word 1–3-grams, TF-IDF and logistic regression, read like D1. Validated first: it must convict `e9_ln` and real URLs whose words have been shuffled (same words, wrong order), and its floors must sit near 0.5 | a validated judge of word order, and a guard for P1 |
| R1-6 | What are the baselines? | property 4 and 5 measures (§4.1) and `uie/gate.py` pass rates, on `e16_b5` and on real data | the reference values P1 is judged against |

**Decision rules, fixed in R1's registration before it runs:**
- **Fields.** A field gets pieces in P1 if its per-field D1 convicts `e16_b5` in either class.
- **Piece count.** 8,000 merges, unless R1-4 shows 32k covers at least 10 more points of
  real words in a field that carries the tell. Then that table uses 16k or 32k, with its
  parameter cost stated before P1 is registered.
- **Go or no-go for units.** If R1-2 and R1-4 show the shortfall lies in words that even 32k
  pieces cannot capture, units cannot plausibly reach the bar. P1 as designed is then not
  worth its GPU cost, and the plan comes back to you with the alternatives (lexicon +
  spelling, or a pretrained prior).

**Cost:** 3–5 h of laptop computing, mostly unattended (learning the pieces is the longest
part, and P1 reuses it). No Kaggle. About 1–1.5 days, including the registration and the
code.

### 5.3 P1 — the representation experiment (Kaggle)

**Arms**

| arm | path, query, fragment | label, subdomains | suffix |
|---|---|---|---|
| **A** | pieces (the path table) | letters | letters |
| **B** | pieces (the path table) | pieces (a host table) | letters |
| `e16_b5` (frozen reference) | letters | letters | letters |

**What is matched between A and B**
- the same rows (`e16_cap`'s) and the same batches (sorted by character length, so
  `e16_cap`'s batch digest holds for both);
- the same start: `e16_cap`'s start plus piece rows from seeded generators, with A's rows a
  subset of B's;
- the same recipe, and the same decode rules plus the piece rule.

A against B therefore isolates the host change. A against `e16_b5` measures the path change
across two trainings; its scale is read against the 0.023 yardstick and the 0.0141
allowance. This reverses Amendment 1's two small choices (long rows, own batches): exact
matching between A and B is worth more than 0.14% of rows.

**Hypotheses**
- **H-A:** pieces in the path, query and fragment bring D1-R to the bar in both classes, with
  new words at the real rate and no cost on the other properties.
- **H-B:** pieces in hosts as well bring the whole-URL D1 to the bar in both classes, with no
  host-specific cost.

**The scorecard**
- **Success (the gold standard):**
  - for A, D1-R CLEARED in both classes and every guard passed;
  - for B, the whole-URL D1 CLEARED in both classes (with D1-R and D1-H) and every guard
    passed.
  - CLEARED means proven below 0.05, the E3 standard, as in Amendment 1. A class counts only
    where the same run convicts `e16_b5`.
- **The continuation rule (whether P2 is worth funding; not a pass mark):**
  - for each reading an arm changes, the share of the distance to the bar it closed:
    (`e16_b5`'s excess − the arm's) ÷ (`e16_b5`'s excess − the clear line).
  - **Promising:** at least 0.5 in a class with every guard passed. **Partial:** 0.2–0.5.
    **None:** below 0.2, or a move within the 0.023 yardstick.
  - **Why it is needed:** without it, "not met" cannot tell "close, continue" from "no
    effect, stop", and the next step would be decided after seeing the data.
- **The guards (failure criteria), against `e16_b5` from the same scoring run:**

| property | guard |
|---|---|
| 1 letters | no gated cell's verdict worsens with x up by more than 0.0141; no PTH, SUB or LBL cell rises by more than 0.0141; FULL up by no more than 0.0141 |
| 3 word order | D4 up by no more than 0.0141 |
| 4 new and not copied | never-seen words not below real by more than T. Exact copies per field not OVER the natural rate, and not further from it than `e16_b5` (queries, paths; for B, labels and subdomains). Strings seen once not copied more than in real data (K17-3). Near misses not above real + T. Registrable-domain copies and threat-snapshot collisions not above `e16_b5` + T (the two-proportion rule) |
| 5 diversity | duplicates not above `e16_b5` + T; distinct words per 1,000 (equal n) not below real − T |
| 6 validity and control | `uie/gate.py` validity pass rate not below `e16_b5` − T; presence within T; all 8 slash rows within T; plan following does not fall; rule integrity has zero violations |

- **The workflow** is gate A's (§15): train, download, laptop VERIFY_TRAIN, generate,
  download, laptop VERIFY_GENERATE, score on a Kaggle CPU session, download, read. Nothing
  is chained.

**Cost**

| where | what | about |
|---|---|---|
| Kaggle GPU | train A and B (two notebooks, may run in parallel) | 2.5–3.5 h each |
| Kaggle GPU | generate A and B, free and control mode, 50 files each | 2–2.5 h each |
| **GPU total** | | **about 10–11 h** |
| Kaggle CPU (no GPU allowance) | score A, B and `e16_b5`: 32 cells, slash rows, word detectors, copying, diversity, validity | 4–6 h of wall time in one or two sessions |
| laptop | preparation, two VERIFY_TRAINs, two VERIFY_GENERATEs, reading | about 10 h, mostly unattended |
| calendar | from P1's registration | 5–7 days |

**A cheaper variant: arm B alone.** About 5–6 GPU-hours.
- **What it still gives:** per-field readings still show whether the path tell and the host
  tell each moved.
- **What it loses:** the clean measure of what host pieces add. And if host pieces fail a
  guard (domain copies, say), there is no host-free arm to fall back on, so a second round
  would be needed.

### 5.4 P2 — what follows P1

| P1's outcome | P2 |
|---|---|
| an arm reaches success | a second training of that arm with another seed, then its full reading. Adoption is your decision |
| promising but not success | the next lever the leftover tell points to: leftover fakes among rare words → lexicon + spelling, or more pieces; a word-order tell → diagnose before choosing; host costs → keep arm A's design |
| partial or none | units are not the lever at this scale. This line stops, and the pretrained-prior route comes to you as a proposal (it needs your approval) |

**Later, and optional:** the two tests `DESIGN.md` names and the programme never ran — a
human sorting test (§1) and detector augmentation (utility). They test the claim from
outside, and are not needed for V2's decisions.

### 5.5 What each stage will teach us

- **R1:** whether the word tell is a problem of common or rare words, in which fields, what
  the fakes look like, how much pieces could fix at most, and whether word order is already
  a tell. P1 then tests the right thing at the right size, or is skipped.
- **P1, A against `e16_b5`:** whether pieces fix the path/query/fragment word tell, and what
  they cost.
- **P1, B against A:** whether pieces in hosts fix the host tell too, and what that costs in
  copying, safety and diversity.
- **P2:** whether a success survives a second training, or which lever the leftover evidence
  points to.

### 5.6 Compared with the current registration

| changed | kept |
|---|---|
| R1 inserted before any training | the step-by-step workflow with laptop checks (gate A) |
| P1 has two arms; hosts are tested in B | scoring on a Kaggle CPU session, with `e16_b5` scored again in the same run |
| success per arm: D1-R for A, the whole-URL D1 for B | the gold standard: CLEARED, proven below 0.05 |
| new judges: D1 per field, and D4 for word order | control mode included; full size (width 512) |
| a scorecard of six properties; copying and diversity guards for hosts | 8,000 merges as the default; the piece rule; the decode rules |
| a continuation rule (not a pass mark) | E3 closed; `e16_b5` frozen |
| exact matching between arms (no long rows; `e16_cap`'s batches) | |
| about +5 GPU-hours | |

### 5.7 Still closed

- decode knobs and sampling changes;
- more steps (R2), and more width or depth (R3);
- new or synthetic training data;
- training against a detector (adversarial or reinforcement);
- filtering or resampling as a generator.

---

## 6. Decisions for the owner

1. **R1 first:** a laptop diagnosis, about 1–1.5 days and no GPU, with its own registration
   before it runs.
2. **P1 with two arms** (A and B, about 10–11 GPU-hours), or **arm B alone** (about 5–6),
   accepting what §5.3 says it loses.
3. **The scorecard:** the gold standard per arm, the six-property guards, and the
   continuation rule.
4. **The V2 finish line (a proposal):** the whole-URL word detector CLEARED in both classes,
   plus E3's two conditions met by a V2 engine. E3 itself stays closed.
5. **The pretrained-prior route** is kept as a later option, brought to you only if units
   fall short.
