# E12 roadmap — query content (DRAFT, not a registration)

> **Decided (2026-09-30).** The owner approved §7's decisions 1, 2, 3 and 5, and
> asked for the copying metric to be checked before deciding 4
> (`DECISION_2026-09-30.md`). The check is done there (§4): the metric is
> query-only, and the treatment in §4.3 awaits the owner's review. QC1 is
> registered next. This draft stays as the planning record.
>
> **QC1 read (2026-09-30, `research/qc1/QC1_RESULTS.md`).** It licenses R1 only.
> Most of this draft's hypotheses were refuted: the tell is not the copy rate, not
> near misses, and not class leakage. It is lexical, not shape. §2.3's bound on
> the length limit ("at most about 0.007") was wrong: QC1-6 measured 0.0163.

- **Status:** a planning draft, 2026-09-30, written after E11's reading at step
  20,000 (`E11_STEP20K_RESULTS.md`, amendment 3).
- **Nothing is registered, coded or run by this draft.** It proposes a laptop
  diagnosis (QC1) and, after it, a Kaggle experiment (E12). Each needs the
  owner's decisions in §7 and its own registration before any code.
- **Built on:**
  - the owner's decisions of 2026-09-28 and 2026-09-29 (`DECISION_2026-09-28.md`,
    `DECISION_2026-09-29.md`);
  - E11 and its two readings (`E11_RESULTS.md`, `E11_STEP20K_RESULTS.md`).
- **Evidence tags:** [CONFIRMED] a registered reading; [MEASURED] a measurement
  made for this draft, on real data only, reported with its script;
  [HYPOTHESIS] consistent with the evidence, not established.

## 0. In plain words

- **What E12 is for.** At step 20,000, E11's plan arm has structure essentially
  right: presence is exact, the slash is in, and every phishing control cell
  clears. What stands between it and E3 is mostly query content. Four of its
  seven open cells are query cells, two more sit in the path-and-query state, and
  three slash rows remain.
- **What this draft measured, on real data only.** Real sites reuse query strings:
  29% of real benign queries and 42% of real phishing ones appear word for word in
  the training data. Our model reuses them far less, 19–23%. It writes too many
  never-seen queries, and too few of the common ones, like `lang=en`, `page=2` or
  phishing's `m=1`.
- **Why that matters twice.**
  - It points at a mechanism [HYPOTHESIS]. A character model spells a common
    string one letter at a time, and every letter is a chance to drift off it.
  - It turns the copying guardrail around. Below the real rate, more copying is
    more realistic, not memorisation.
- **The proposal: diagnose first.** QC1 runs on the laptop only, from the
  step-20,000 files we already have. It asks where the tell is:
  - in the mix of common and new strings;
  - in the content of the new ones;
  - in phishing borrowing from benign;
  - in the length limit.

  Each answer licenses one fix for E12's Kaggle run:
  - subword tokens for the query, the project's original design;
  - the class read at every position;
  - a longer length limit.
- **What does not change.** No capacity, no more steps, no decoder changes. The
  gate and its thresholds stay as they are.

## 1. Where E3 stands after E11

`e11_plan` at step 20,000 [CONFIRMED, `E11_STEP20K_RESULTS.md` §5, §9]. Every
other gated cell, free and control, is CLEARED.

| mode | cell | x | verdict | at step 14,000 |
|---|---|---|---|---|
| free | benign QRY | 0.0350 | UNDECIDED | 0.0387 UNDECIDED |
| free | phishing QRY | 0.0953 | **CONVICTED** | 0.1382 CONVICTED |
| control | benign P0Q1 QRY | 0.0581 | UNDECIDED | 0.0847 CONVICTED |
| control | benign P1Q1 QRY | 0.0511 | UNDECIDED | 0.0745 CONVICTED |
| control | benign P1Q1 PTH | 0.0509 | UNDECIDED | 0.0578 CONVICTED |
| control | benign P1Q1 SUB | 0.0379 | UNDECIDED | 0.0513 UNDECIDED |
| free | phishing PTH | 0.0431 | UNDECIDED | 0.0533 UNDECIDED |

Plus 3 of the 8 slash rows: phishing bare hosts in both modes (99.53% and 99.64%
against 100%), and benign host-route queries in control mode (99.97% against
99.551%).

- **Four of the seven are query cells,** and two more are in the path-and-query
  state, the URLs that carry queries. Query content is E12's target.
- **The benign query cells sit near the bar** (0.035–0.058). Phishing's (0.0953)
  is far from it, though it fell by a third at step 20,000.
- **Phishing free PTH** is the long-standing phishing path cell. Its attribution
  branch stays closed (§8).

## 2. What we know about query content going in

### 2.1 The old diagnosis, on E5 [CONFIRMED, `research/qry/`]

- **The tell is in the query's shape.** Replace every letter with `a` and every
  digit with `0`, keeping punctuation and lengths, and the detector still tells
  generated from real just as well (`skeleton`, Q3 and Q3.1; phishing the same,
  `PHISHING_QRY_RESULTS.md`).
- **It is spread across properties; none carries half.** Benign: parameter names
  17–28%, parameter count 11–33%, long values 3–6%. Phishing: digit-against-letter
  layout 21–32%, names 16–27%, value composition 11–12%, count 6–13%, long values
  6–10%.
- **The model concentrated on few parameter names.** At equal sample size, 36
  distinct names per 100 parameters (46 with the temperature decoder), against
  real's 60.
- **This was measured on E5,** two objectives and three structure changes ago:
  - E9's length-neutral objective;
  - E10 and E11's presence plan;
  - E11's root slash and clipping repair.

### 2.2 N2a, on E7 [CONFIRMED, `N2A_RESULTS.md`]

- **Four times the query practice changed nothing the detector sees.** The model
  trained on query rows at 4× E7's exposure (A2) was as detectable as E7:
  - against held-out queries: +0.0945 against +0.0957;
  - **against its own training queries: +0.0839 against +0.0862.**
- **What it gained was copying:** 34.2% of its benign queries were training
  queries word for word, against E7's 27.3% (phishing 36.1% against 26.4%).
- **The registered reading was SPARSITY** (MIXED under the plan's original
  wording). Its consequence: no training repair of query content licensed; R3
  locked.
- **The strongest fact here is the line in bold.** Even against queries it had
  seen about eight times, the model's output is told apart. The limit is how the
  model writes queries, not only how many it has seen.

### 2.3 Measured for this draft: real sites share query strings [MEASURED]

Source: `research/native/e12_sizing.py` → `data/e12/sizing.json` (real data only;
no model output read). Pools as the gate's free-mode QRY cell: `name_val`, one URL
per host, the gate's seed.

| class | real queries in the pool | found word for word in the training queries | median length, shared / never seen |
|---|---|---|---|
| benign | 7,253 | **29.3%** | 7 / 25 characters |
| phishing | 1,068 | **41.9%** | 8 / 31 characters |

- **The shared strings are the short, common head:**
  - benign: `lang=en` (1.2% of all real queries), `page=2`, `page=1`, `page=3`,
    `p=1`, `amp=1`, `id=1`;
  - phishing: kit artefacts. `m=1` alone is 6.9% of real phishing queries; one
    UUID recurs on 16 held-out hosts and in 209 training rows; the corpus's
    masked-email placeholder `email=3mail@b.c` is common.
- **This is the natural copy rate.** A generator that sampled the real
  distribution would reproduce training queries at exactly these rates, because
  a held-out real URL is itself such a sample.
- **Checked since (`DECISION_2026-09-30.md` §4.2):**
  - benign's rate is entirely reuse between different sites; the split is by
    registrable domain;
  - phishing's split is by host, and on its fully domain-disjoint hosts the rate
    is 26.5%;
  - matches to strings seen only once in training are 4.6% and 5.1% of the pools.
- **E11's arms copy far less.** K11d's rates (both classes pooled) were 17.7–23.0%
  across arms and modes at `best.pt`, and 19.5–23.0% at step 20,000
  (`E11_STEP20K_RESULTS.md` §6). Weighted by `e11_plan`'s own free-mode query
  counts, the natural rate would be 34.5%.
- **So the generator writes too many never-seen queries,** not too few. It
  under-produces the common exact strings that real URLs reuse.

**The length limit** [MEASURED, same script]. The share of real query URLs whose
encoding exceeds 256 tokens, so the generator can never write them:

| class | training rows with a query | the gate's real pool |
|---|---|---|
| benign | 1.51% | 1.43% |
| phishing | 3.92% | 3.56% |

At most, these could account for about 0.007 of benign's tell and 0.018 of
phishing's (half the share, if the detector caught every one).

### 2.4 What the new facts change

- **The copying guardrail points the wrong way below the natural rate.**
  - K11d (and DC1, N2a before it) treats any rise in copying as memorisation.
  - Up to 29% (benign) or 42% (phishing), a rise is movement toward the real
    distribution.
  - E9's objective lowered copying by 7.97 points in free mode: 27.23% → 19.26%,
    both classes pooled (`E9_RESULTS.md` §8). For those query counts the pooled
    natural rate is about 34%. Both were below it, and the fall moved further
    away. It was read as an improvement at the time.
  - E11's K11d stays read as registered. The proposal (§7, decision 4) is for E12
    onward: judge copying against the natural rate, in both directions.
- **N2a's copying clause lacked this reference.**
  - Phishing A2's rise (26.4% → 36.1%) was still below the natural 41.9%.
  - Benign A2 (34.2%) went 4.9 points above the natural 29.3%.
  - N2a's reading stays as recorded. The point is only that its copying evidence
    reads differently against the natural rate, and that the fit evidence (A2
    as detectable against its own training queries) is the part that stands.
- **The head matters, not only the tail.** E5's diagnosis looked at what the
  model fails to invent (rare names, long values). The shared-string gap says it
  also fails to reproduce what real sites repeat.
  - [HYPOTHESIS] A character model gets many chances to go wrong while spelling
    a common string. `lang=en` is seven sampled characters, and a slip on any one
    turns a real string into a never-seen one. That would produce both symptoms:
    fewer exact common strings, and more slightly-off novel ones.
  - The project's own design anticipated this. `DESIGN.md` §4.2 specified
    subword tokens for paths, queries and fragments: "a valid pattern becomes one
    atomic unit instead of three independent chances to go wrong". E2 built a
    character tokenizer because it was cheap (commit 079592f). No later document
    argues against subwords.

### 2.5 The instrument's power

- **The free-mode QRY cell compares few queries.** It uses the generated URLs that
  carry a query: about 530 per generation (benign) and 375 (phishing). Control
  mode's query cells compare 1,200.
- **A detector trained on fewer examples scores lower.** So free mode's cell is
  the weaker instrument, and it moves with the query rate:
  - E7's free benign QRY read UNDECIDED at n = 289, and CONVICTED at +0.0957 at
    full n (DC1);
  - N2a read it at n = 1,200.
- **Consequence for E12:** content progress is read at a fixed n as well as on
  the gate. The gate stays unchanged, and E3 is still judged on it.

## 3. The proposal in one line

**Diagnose first, on the laptop, from E11's existing step-20,000 generations
(QC1). Then train only the levers the diagnosis licenses (E12).**

Why not go straight to Kaggle:
- The last content decision (N2a) licensed no training lever, and §2.3 changes how
  part of its evidence reads.
- The model has changed too much since E5 for the old ladder to pick a lever.
- A Kaggle round costs the owner about 12 hours. QC1 costs laptop hours only,
  and it reads 720,000 URLs that already exist.

## 4. QC1 — what gives the queries away (laptop diagnosis)

**Material:**
- `e11_plan` at step 20,000 (the base), both classes: free mode, and control mode's
  query states;
- `e11_ref` at step 20,000 as the replication model: a second training with the
  controls hidden;
- the gate's real pools and the training queries (`dc1_compare.training_queries`).

**Instruments:** existing ones only. The gate's detector with its real-against-real
floor, applied through `q3_content.scored` (the Q3 harness), and Q3.1's transforms
(`qtransforms.py`). No model is trained and nothing is generated.

**Readings** (the registration fixes every number):

| code | question | how | reading |
|---|---|---|---|
| QC1-1 | Does the model copy at the natural rate? | generated copy rate per class, mode and control state, each against its own real pool's natural rate (phishing at the host level and domain-disjoint), with the detectability tolerance; the match-band profile, including strings seen once in training (`DECISION_2026-09-30.md` §4) | UNDER / WITHIN / OVER |
| QC1-2 | Is the tell in the mix, or in the content? | the gate's excess (i) as the gate reads it, and (ii) with the generated set re-drawn so its shared share equals the natural rate (a diagnostic re-draw, never a generator) | MIX-LED if (ii) is at most half of (i); CONTENT-LED if (ii) keeps at least 0.8; else SPLIT |
| QC1-3 | Which properties carry the tell? | Q3.1's ladder on the gate's populations, so it compares with E5's reading; two stages, promoted only if it replicates on `e11_ref` | IDENTIFIED / DISTRIBUTED, and whether `skeleton` keeps the tell |
| QC1-4 | Are the novel queries near misses? | the share of never-seen queries one edit from a frequent training query, generated against real | BLENDING if generated exceeds real by the registered margin |
| QC1-5 | Are phishing queries pulled toward benign? | the excess of generated phishing queries against real benign ones, against the same for real phishing queries | LEAK if the generated ones sit measurably closer to benign |
| QC1-6 | How much does the length limit account for? | the gate's excess with the real pool restricted to URLs that fit 256 tokens | reported, as a share of the tell |

**The decision table** (fixed in QC1's registration; each reading licenses a lever
family for E12, never a fix by itself):

| QC1 reading | E12 lever it licenses | prior arguments, stated |
|---|---|---|
| MIX-LED, or BLENDING | **query subwords:** `DESIGN.md` §4.2's subword tokens for the query only (paths and everything else stay as they are) | none recorded against; the largest code change (tokenizer, loss constant C, the shared start) |
| LEAK (phishing) | **class conditioning at every position:** the class embedding added to every input position, not only read from the first token | none recorded against; small code change |
| the length limit accounts for at least a quarter of a class's tell | **R1:** `max_len` 384 | `REPAIR_PLAN.md` R1: optional, "only if a training run is already happening" (E12 is one); 2.25× attention cost |
| CONTENT-LED and DISTRIBUTED, with no BLENDING or LEAK | none | the owner decides. The recorded route to capacity is N2a re-measured on the current objective (`DECISION_2026-09-29.md` §4) |

- **Not licensable by QC1:**
  - a query-content loss weight (N2b). N2a did not license it. §2.4 bears on
    N2a's copying clause, so reopening it is the owner's call;
  - more steps (R2, rejected);
  - capacity (R3, locked);
  - any decoder change. The owner's rules exclude masking as the core mechanism.
    R4 also showed truncation concentrates names.

**Cost:** laptop only. About 12 detector fits per reading and stratum; the ladder
is about 17 transforms × 12 splits × 2 stages. [HYPOTHESIS] about one working day
with the pool runner.

## 5. E12 — the Kaggle experiment, registered after QC1

- **Arms,** from one start, in one session:
  - `e12_base`: `e11_plan`'s recipe exactly (the slash, the repair, the plan; the
    supplied plan in free mode);
  - one arm per licensed lever, each `e12_base` plus that lever. At most three.
- **Why a same-session base.** A second training of the same recipe differs by
  GPU nondeterminism. Every lever is measured against a base trained alongside it.
- **The start.** E11's method (D2): new token rows are drawn from a separate seeded
  generator, and the rest equals the recorded start.
- **The checkpoint.** `final.pt`, step 20,000, if the owner adopts it (§7).
- **Generation.** E11's 50 jobs per arm (E11's 200 were over its four arms;
  corrected 2026-09-30, before any E12 data).
- **Criteria** (sketch; the registration fixes them):

  | code | criterion |
  |---|---|
  | K12a content | each targeted QRY cell falls by more than the retrain allowance (0.0141) against `e12_base`, at the gate's n and at a fixed n |
  | K12b structure | K11a and K11b as `e12_base` reads them: no new failure |
  | K12c no regression | K11c's rule against `e12_base` |
  | K12d guardrails | novelty as K11d; copying two-sided against the matched natural rate, and a one-sided guard on copying strings seen once in training (`DECISION_2026-09-30.md` §4.3, awaiting the owner's review) |
  | reported | the E3 finish line for any candidate; QC1's readings again on each arm |

- **Cost:** Kaggle training about 1 h per arm (subwords shorten sequences;
  `max_len` 384 lengthens them); generation about 1.6 h per arm; laptop about 3 h.

## 6. Honest expectations

- **QC1 may read DISTRIBUTED again,** as E5 did. Then no lever is licensed, and
  that is itself the finding: the character model's limit would be general, and
  the next step is the owner's.
- **Subwords are the most promising lever, and the most expensive.**
  - They make common strings atomic, which should fix the head.
  - They do nothing for a tail the model never learned.
  - They add their own risk: the model can over-reproduce frequent tokens. The
    natural-rate copying guard would catch that.
- **Phishing is the hardest class.**
  - Its tell barely moved for most of the project: E5 +0.143 to +0.149 on the
    gate, E11 +0.1382 at step 14,000.
  - At step 20,000 it fell to +0.0953, still CONVICTED and far from the bar.
  - It has a tenth of benign's data.
  - It also leaks in structure: phishing bare hosts miss the slash, which no real
    phishing host does. LEAK is the reading I expect there [HYPOTHESIS].
- **Nothing in E12 targets paths.** Phishing PTH and benign P1Q1 PTH (§1) keep
  their existing rules (§8).

## 7. The owner's decisions before QC1's registration

1. **The base.** `e11_plan`'s recipe (the slash, the clipping repair, the presence
   plan, the supplied plan in free mode) as E12's base. At step 20,000 it is exact
   on presence, the best arm on content, and clears every phishing control cell
   (`E11_STEP20K_RESULTS.md` §10). Joint dropout is left out: at the same step it
   changes nothing that matters.
2. **The checkpoint.** Read `final.pt` (step 20,000) in every later registration.
   The owner deferred this until the step-20,000 reading was in. It is now:
   - the step changed four of `e11_plan`'s verdicts, all toward E3;
   - it removed every K11c regression;
   - it shrank every attribution.
3. **Order.** QC1 on the laptop first (recommended), or E12 directly with arms
   chosen now.
4. **Copying.** From E12 on, judge copying against the natural rate in both
   directions, instead of "must not rise". E11's K11d stays as read.
5. **Scope.** Query content only (recommended), or also the slash residuals and
   benign P1Q1's path and subdomain cells.

**Outcome, 2026-09-30** (`DECISION_2026-09-30.md`):

| decision | outcome |
|---|---|
| 1. base | approved: `e11_plan`'s recipe; E11 stays the official baseline |
| 2. checkpoint | approved: `final.pt` for every arm; `best.pt` only with the owner's explicit approval |
| 3. order | approved: QC1 first, on the laptop, before any E12 training |
| 4. copying | the metric is query-only and the four checks pass, with two qualifications (§4.2). The treatment (§4.3) awaits the owner's review |
| 5. scope | approved: query content only. The slash residuals and benign P1Q1 PTH and SUB stay documented residual limitations |

## 8. Not in E12

- **Phishing PTH:** the attribution branch stays closed (owner). Its cells are read
  at every step, as before.
- **Benign P1Q1 PTH and SUB:** documented residual limitations (owner, §7,
  decision 5). The native plan requires a registered diagnostic before any repair
  of control-state path cells.
- **The slash's residual rows** (K11b): documented residual limitations (owner,
  §7, decision 5).
- **Capacity (R3), more steps (R2), decoders, rejection, masking, second passes,
  hybrids:** excluded by the owner's rules.
- **E3's gate, thresholds and published results:** unchanged.
