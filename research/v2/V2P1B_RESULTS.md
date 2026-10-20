# V2-P1b — results: stopped at K-used (MEMORY UNUSED, by the registered rule)

- **Registration:** `research/v2/PREREG_V2_P1B.md` (d3561a7; Status 947a8e9), with P1's
  `PREREG_V2_P1_MEMORY.md` (904a79a; amendment 1, 81060e8).
- **Records:** `data/v2/p1b/`: `train/`, `VERIFY_TRAIN.json`, `k_used.json`,
  `checks_job.log`, `diag_frames.json`, `diag_frames_job.log`.
- **Read once, 2026-10-13.** VERIFY_TRAIN passed first.
- **Generation did not run.** That is P1's §6 stop: K-used failed.
- **Nothing is adopted.** `e16_b5` stays the baseline.

## In plain words

- **The run is valid, and the fix worked.** P1's BatchNorm fault is gone: the engine
  computes the same thing in training and in generation. All three audits pass, and
  VERIFY_TRAIN passes.
- **The memory is used heavily.** It chooses 89.6% of its slots. Switching it off costs
  0.58 nats per character on benign and 0.78 on phishing.
- **K-used fails on benign.**
  - Benign: p1b_mem is 0.014 nats per character worse than `e16_cap`, interval
    [+0.009, +0.020]. The bar needs the whole interval below 0.
  - Phishing: 0.019 better.
- **Where the benign loss comes from (post hoc diagnosis, §4):** one place only, domain
  names on page rows.
  - Page rows are the structure corpus, where every site appears with several pages.
  - On pages of sites the engine trained on, the memory makes domain names easier to
    predict. On pages of new sites it makes them much harder: +0.335 nats per character.
  - **It learned to recall site names.** That recall misleads it on every site it has not
    seen. Validation sites are not training sites: the benign split is by domain, and the
    phishing split by host.
- **Everywhere else the memory helps, on new sites too:**
  - paths and queries in both classes, for example phishing queries −0.12 to −0.15 nats per
    character;
  - domain names on name rows, where each name appears once (−0.03 to −0.04).
- **The judged fields improved, but only a little.** V2's word judge D1b-R replaces the
  host with `h.h`, so the failing field is not the one it reads; paths and queries are. The
  gains there are about 2% (benign paths −0.02 to −0.035 nats per character), the size of
  P1's in-conditions gain and of E16's width step. A 30% cut in the word tell (K2) does not
  follow from gains this size.
- **The failing behaviour is one the owner ruled out.** Condition 2 forbids a memory that
  memorises registrable domains. Recalling training site names is the start of that.

## 1. Validity: VERIFY_TRAIN PASS (laptop, 2026-10-13, 8.6 min)

| check | result |
|---|---|
| V1 | `final.pt` equals the checkpoint training recorded (sha256). The code is unchanged since `abd3e02`: only the registration file changed |
| V-iso | `uie/` equals `422cb23` |
| V-recipe | `e16_cap`'s settings exactly, 2 parts. Memory: query LayerNorm, no branch dropout, values' own Adam group at 1e-3. 161,792,000 parameters |
| V-start | `e16_cap`'s part `caca0107…`; the memory's start as recorded before training |
| V-batch, V-stream, V-mass | `adbddc2a…`; `637accab…`; the injection equals `e16_cap`'s exactly |
| K12e | 0% clipped after step 1,000 in every band |
| V-audit (T4) | Kaggle before training: largest gap 0.0000015 logits. In the run, on the exact model before step 1: 0.0035. On the laptop, on the trained `final.pt`: 0.000018. No stored statistic changed in any of them. Bar 0.05 |
| tests | `test_v2_mem`, `test_v2_p1b`, `test_e16`, `test_e17`: exit 0 |
| generation check | 9 URLs on the laptop CPU, all parse (`snargudaryan.github.io/netflix-clone`, `myclinic.cc/en/product/red-thoughts-hour/?id=Y`, …) |

**The configuration as trained:**
- `e16_cap`'s recipe: width 512, 8 layers, 20,000 steps, batch 256 in 2 parts, rate 3e-4,
  stream 1. The memory sits after block 4: 262,144 slots (512 × 512 product keys), 4 heads,
  32 slots per head per character.
- P1b's settings: T1 (no dropout on the memory's output), T2 (the values in their own Adam
  group, peak 1e-3, no weight decay), T4 (the audit). No tripwire (T3 no).

**Resources:**
- Kaggle T4: the training loop took 4.12 h (`e16_cap`'s: 2.73 h); the notebook ran 257 min.
- Peak GPU memory was 11.2 of 14.6 GB, in 2 parts.
- Generation never ran, so about 3.5 GPU-hours were not spent.
- Laptop: the checks took 23 min, and the diagnosis (§4) 36 min. Both ran at low priority
  with the low-RAM alarm, which never fired.

## 2. K-used (P1 §5.3): FAIL

3,991 benign and 3,992 phishing validation rows (P1 §4.6), in nats per content character:

| | `e16_cap` | `p1b_mem` | difference [95% interval] | ablation (memory off) |
|---|---|---|---|---|
| benign | 1.697 | 1.711 | **+0.0140 [+0.0086, +0.0195]** | +0.577 |
| phishing | 1.701 | 1.682 | −0.0186 [−0.0254, −0.0116] | +0.778 |

- **(a) fit:** FAIL, since benign's interval lies above 0.
- **(b) usage:** PASS: 234,763 of 262,144 slots (89.6%), KL from uniform 0.99. P1 had
  84.4% and 2.30.

**By field** (`p1b_mem` minus `e16_cap`):

| field | benign | phishing |
|---|---|---|
| suffix | −0.002 | −0.001 |
| domain name (LBL) | **+0.139** | +0.022 |
| subdomain | +0.054 | +0.023 |
| path | −0.026 | −0.040 |
| query | −0.039 | **−0.141** |
| fragment | – | −0.076 (7 rows) |

## 3. The training record, reported

Validation loss as training measures it (all validation rows, the training objective).
Negative means `p1b_mem` is better:

| step | 1,000 | 3,000 | 5,000 | 6,000 | 7,000 | 10,000 | 15,000 | 20,000 |
|---|---|---|---|---|---|---|---|---|
| benign: `p1b_mem` | 1.985 | 1.649 | **1.554** | 1.558 | 1.565 | 1.597 | 1.623 | 1.628 |
| benign: `e16_cap` | 2.016 | 1.691 | 1.577 | 1.559 | **1.551** | 1.570 | 1.603 | 1.607 |
| benign difference | −0.030 | −0.042 | −0.023 | −0.001 | +0.014 | +0.028 | +0.020 | **+0.021** |
| phishing difference | −0.044 | −0.041 | −0.040 | −0.035 | −0.033 | −0.035 | −0.029 | **−0.027** |

- **Benign:** `p1b_mem` led until step 5,000, crossed near 6,000, and ended 0.021 behind.
  Both engines overfit benign, `e16_cap` after step 7,000 and `p1b_mem` earlier and more.
- **Phishing:** `p1b_mem` led throughout.
- **The run used `final.pt` (step 20,000),** as the owner requires. `best.pt` was not
  extracted.

## 4. The diagnosis (post hoc: `v2_p1b_diag_frames.py`, committed 8f2729b before it ran; decides nothing)

**Why it was needed.** The per-field table that training printed on Kaggle put `p1b_mem`
0.044 BETTER than `e16_cap` on benign domain names. K-used put it 0.139 WORSE. The two
differ in rows, frame and device. Three readings separate them.

**B: the device is not the cause.**
- Training's own instrument (`field_nll`) on training's own rows (the first 3,000 name rows
  per class, domain frame) was rerun on the laptop CPU.
- It reproduces the Kaggle GPU table to 4 decimals in every field (benign domain names
  −0.0436, phishing queries −0.0957, …).
- That table covers name rows only, which is why it disagreed with K-used.

**A: K-used's own rows, split by corpus** (`p1b_mem` minus `e16_cap`, with 95% intervals):

| field | benign, name rows | benign, page rows | phishing, name rows | phishing, page rows |
|---|---|---|---|---|
| all | −0.034 | **+0.046** | −0.048 | −0.001 |
| domain name | −0.043 [−0.050, −0.035] | **+0.335 [+0.298, +0.369]** | −0.027 | **+0.070** |
| subdomain | −0.002 | **+0.100** | −0.056 | **+0.131** |
| path | −0.035 | −0.022 | −0.065 | −0.034 |
| query | −0.050 | −0.033 | −0.122 | −0.146 |
| memory off: domain name | +0.570 | **+1.147** | +0.982 | **+1.126** |

- The two corpora add back to K-used exactly: benign domain names +0.1386, against K-used's
  +0.1387.
- **The whole benign loss is on page rows, in domain names and subdomains.** It is also
  where the engine leans on the memory hardest: switching it off costs 1.15 nats per
  character there.

**C: page rows from sites the engines trained on** (2,000 per class from structure_train):

| field | benign: trained sites | benign: new sites (A) | phishing: trained sites | phishing: new sites (A) |
|---|---|---|---|---|
| domain name | **−0.025** | **+0.335** | **−0.167** | **+0.070** |
| subdomain | −0.122 | +0.100 | −0.262 | +0.131 |
| path | −0.086 | −0.022 | −0.138 | −0.034 |
| query | −0.145 | −0.033 | −0.167 | −0.146 |

- **The memory makes recalled site names cheaper and new site names dearer.** That is the
  signature of memorisation.
- `e16_cap` already memorises them. Its benign page-row domain names score 0.60 nats per
  character on trained sites and 3.23 on new ones. The memory widens that gap, to 0.58
  against 3.56.
- Paths and queries also gain more on trained sites than on new ones, but the gain on new
  sites stays: they share pieces across sites.

**What it means.**
- The memory is a working component. It learned the parts of URLs that recur, which is
  what a memory is for.
- In the page corpus, the most recurrent string of all is the site's own name, repeated on
  every page. The memory stored those names, and that hurts every new site.

## 5. Predictions [HYPOTHESIS] against the outcome

| reading | prediction | outcome |
|---|---|---|
| the audit on Kaggle | PASS (about 98%) | PASS |
| K-used | PASS (about 70%); fit at least `e16_cap`'s; paths −0.02 to −0.06 | **FAIL.** Paths −0.026 (benign) and −0.040 (phishing), as predicted. The fit is worse on benign |
| K2 | benign about 25%; phishing about 10% | not reached |
| SUCCESS | about 5% | not reached |

## 6. Deviations from the registration

- **Amendment 1 was logged and withdrawn before any training output existed.**
  - It proposed running VERIFY_TRAIN and K-used on Kaggle (0b9f6ae, revised in 24c19e7).
  - The owner withdrew it ("you wont build it right , forget it leave everything before I
    tell you to combine them …"). The revert 6a6509f restored the registration exactly.
  - Nothing of it was built or run. The checks ran on the laptop, as §7 registers.
- **The diagnosis (§4) is post hoc.** It decides nothing.
- Nothing else.

## 7. What follows (the owner decides; nothing runs before that)

1. **Close the memory route (my recommendation).**
   - Two runs have now tested it. The memory works mechanically, but on the fields the
     word judge reads it buys about 2%, the size of P1's in-conditions gain.
   - A 30% cut in the word tell does not follow from that.
   - Its largest learned behaviour, recalling training site names, is what condition 2
     forbids.
2. **P1c: a memory kept away from domain names,** for example read only inside paths and
   queries.
   - It would remove the site-name recall and keep the gains on the fields D1b-R reads.
   - It is an architecture change: a new registration and your approval, then about 4 h of
     training, 3.5 h of generation and a day on the laptop.
   - The expected gain is the same small one.
3. **Generate with this `final.pt` anyway,** for E3's last weak cell. Phishing queries had the
   memory's largest gain (−0.12 to −0.15).
   - I advise against it. It would override a registered stop on a reason found after
     seeing the data.
   - The cell was far from its bar (0.1141 in E12).
   - The site-name recall puts the domain copy guards and the host-word checks at risk.

**If the route is closed:** the owner's draft-and-revise idea, deferred until after P1b, is
the open decision.
