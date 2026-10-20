# Option A's data — a second stream of new benign query rows (acquisition only)

- **Registered:** 2026-10-06, before the crawl runs.
- **What this is:** the data for option A (a second doubling of E15's stream), acquired
  now, while E16 trains on Kaggle. **It is not an experiment.**
  - Nothing is trained or generated with these rows.
  - They are held unused until an E17 registration is approved by the owner.
- **Authorization:**
  - `PREREG_E16.md` §13.1, preparation: "I explicitly authorize you to prepare the script
    for the Option A crawl … We will sequence the background crawl after Kaggle is
    successfully running";
  - the run, after `E16_train` was confirmed running: "go for oth pls" (both: the laptop
    scripts and the crawl).

## 1. The procedure: E15's, unchanged except where listed

The procedure is `PREREG_E15.md` §§9.4, 11.2 and `research/e15/e15_acquire.py`:
- corpus2's 8 live crawls (CC-MAIN-2024-42 … 2026-30);
- `uie/corpus/extract.py` (suffix-stratified blocks, alpha 0.5, status 200 + text/html, up
  to 48 URLs per domain per file);
- `merge.py`'s name rule (one URL per registrable domain, reservoir k = 1);
- the exclusions and filters, and the per-stratum cap by a hash order.

| | stream 1 (E15, in `e15_new`) | **stream 2 (this)** |
|---|---|---|
| rounds | 1–8 | **9 onward**: the same block-seed formula, 20261005 + 7919·i + 1,000,003·(r − 1), so every round draws new blocks |
| budget per crawl | 4,251 blocks (corpus2's own); 4,000 used | **its own 4,251**: rounds 9–16 at 500, then round 17 at 251. Stops early once both targets are met |
| reservoir seed | 20261006 | **20261013**, over this stream's shards only |
| exclusions | `split_random`'s domains; the phishing feed's hosts and domains (`feed_index.json`, frozen at d0642d7) | the same, **plus every registrable domain in stream 1** (`qry_extra.jsonl`, 57,292 rows, `637accab…`), so no domain appears in both streams |
| filters | a query; P0Q1 or P1Q1 by presence; at most 384 tokens as encoded | the same |
| targets | N_old per stratum: P0Q1 14,863, P1Q1 42,429 | the same: one more doubling |
| order and cap | sha256("E15-B:" + domain), first N_old | sha256("**E17-A:**" + domain), first N_old |
| concurrency | at most 4 crawls in parallel, 4 attempts, 180 s pause (E15's lesson: Common Crawl throttles 8) | the same |
| output | `data/e15/qry_extra/` | **`data/e17/qry_extra2/`**: shards, logs, the two stratum files, manifest. Stream 1's folder is not touched |

## 2. Expected yield, and the shortfall rule

- **Expected:** E15 found 16,028 P0Q1 and 47,692 P1Q1 rows in 32,000 blocks. New blocks
  partly revisit known domains, and stream 1's domains are now excluded. So I expect about
  65–90% of that: P0Q1 about 10,500–14,500 and P1Q1 about 31,000–43,000. **A shortfall is
  likely, mostly in P0Q1.**
- **If a stratum is short after the budget:** all its rows are kept. An E17 registration
  would inject it at p = N_new / N_old, E15's own rule (`PREREG_E15.md` §10.4). The budget
  is not extended without the owner.

## 3. What is not done here

- **No population gates yet** (E15's P1–P8). They are run on stream 2, against old +
  stream 1, as the first step of any E17 registration, before training.
- No training, no generation, no use of the rows.
- No change to E16, whose data stays `e15_new`'s.

## 4. Change, 2026-10-06 20:45: patient retries for truncated reads (operational)

- **What happened.** Round 9's first attempt lost all 8 crawl workers between 20:24 and
  20:39, each to `http.client.IncompleteRead`: a response cut off mid-body.
  - The library's `_get` (`uie/corpus/commoncrawl.py`) retries three times at once, and
    does not catch this error at all. So one truncated response ended a whole 500-block
    worker.
  - The workers died at different times, so this was not one network outage. Common
    Crawl was cutting responses.
- **The change.** In the crawl workers only, `_get` is replaced by `patient_get`
  (`e17_acquire.py`):
  - the truncated-read error is retried as well;
  - up to 8 attempts, pausing 10 s, then 20 s, doubling, at most 300 s.
- **Why it changes nothing in the data.** A retry fetches the same byte range again, and
  only a complete response is returned. So every block and every row is what the
  library's own call returns when nothing is cut off. Nothing else changes.
- **Not changed:** `uie/`. A change there would also trip E16's own check that the
  library is unchanged since its GPU run (V1).
- **Round 9 restarts from scratch.** No shard of it was finished (no `.done.json`).
  Rounds, seeds and budget are as in §1.

## Status

    2026-10-06  registered (this document and research/e17/e17_acquire.py), before the
                crawl runs.
    2026-10-06  round 9 failed on truncated reads; stopped. Section 4: patient retries in
                the workers only; the data drawn is unchanged. Restarted.
    2026-10-07  COMPLETE after round 16 (8 rounds x 8 crawls x 500 blocks = 32,000
                blocks; 17,192,353 rows read, 1,056,382 domains; 2,669 domains dropped
                as stream 1's). Both strata met: P0Q1 14,863 of 15,893 available
                (benign_P0Q1.jsonl, sha256 23c64419096a04d4a35042094320ee57ae9392d547e6a3a1fd3168f0e069b698);
                P1Q1 42,429 of 46,790 (benign_P1Q1.jsonl, sha256
                a153e75eefd348b107d3a89e32a1292df12674aaa73510bba4a90c9bf7a6b1de). No
                shortfall. Held unused; gates P1-P8 are the first step of any E17
                registration. Data in data/e17/qry_extra2 (not in git).
