# V2 — the made-up words: is draft-and-revise the answer? An investigation

- **The question, 2026-10-13:** evaluate the draft-and-revise idea; search the research
  literature for this problem; present the solutions found; recommend the best one and say
  exactly why.
- **Not a registration.** It decides nothing and changes nothing. `e16_b5` stays frozen, and
  nothing was trained or generated.
- **New evidence gathered for it:**
  - **A probe of the idea** on the frozen engine (`research/v2/v2_dr_probe.py`, committed
    25c634a before it ran). Output: `data/v2/dr_probe/probe.json`. Inference only, 3.3 min
    on the laptop CPU.
  - **A literature search** (sources in §7).
- **Earlier evidence used:** R0, R1, R2, R3, P1 and P1b (`research/v2/*_RESULTS.md`).

## 0. In plain words

1. **The engine mostly does not misspell. It does not know enough rare real words, so it
   makes new ones up.**
   - In benign paths, only about a third of the excess made-up words are misspellings of
     common words.
   - Fixing every misspelling perfectly would close only 27% of the word tell (R2's oracle).
     The rest are inventions and glued words, standing in for real words the engine never
     learned.
2. **Your idea cannot fix that, and the probe shows why on our own engine.**
   - **The detector fails where it matters.** In benign paths, the engine is just as
     unsure when it writes a *real* rare word as when it invents one. Flagging half of its
     made-up words would also flag 47% of the real rare words, which are already in short
     supply.
   - **The reviser would make it worse.** Given a real rare word next to a far more common
     look-alike, the engine prefers the common look-alike 69% of the time. A reviser using
     the engine's own judgement would turn rare real words into common ones: the opposite
     of what is needed.
   - **The deeper reason:** revising cannot add knowledge. A second pass with the same
     knowledge either changes nothing or trades made-up words for common ones. The research
     literature finds the same: models cannot correct themselves without new information.
3. **Your intuition about *where* the problem is was right.** The trouble sits at the
   uncertain spots, where a rare word should begin. The fix is to give the engine word-level
   knowledge *before* it writes, not a second pass afterwards.
4. **The best-evidenced solution: write paths, queries and fragments in word pieces (BPE).**
   That is V2's original design, which you approved and which was never trained.
   - **Corrected in §8, at the owner's question:** the host must be tested too (arm B),
     with a piece table built so that it cannot hold whole sites.
   - On words seen 1–99 times in training, our exact problem band, a published comparison
     finds word pieces almost as good as a full word list (1.70 against 1.64 bits per
     character). Letter-by-letter writing scores 2.08.
   - On multilingual text, word pieces beat the word list in all seven languages tested.
     URLs are multilingual.
   - Pieces are learned fragments, not a word list: the engine still writes freely, piece
     by piece, and can still coin new words.
5. **I was wrong about pieces before.** In R1 I argued that pieces cannot reach rare words,
   because few rare words are held whole by a single piece. That was a proxy, and R1 flagged
   it as one (its L2). The published comparison shows pieces help rare words even when
   they are split.
6. **Honest odds: a real improvement is likely; the gold standard is not.** The bar (an 81%
   cut in benign) is far. No method measured so far has come close without copying.

## 1. What the problem is, measured

**The tell lives in the words, not the shape** (R2: vocabulary features alone give AUC
0.657 on benign paths; structural features alone 0.526).

**Benign paths, share of words by how often training saw them** (this probe, 1,500 URLs per
source; R1 and R2 agree):

| | never | 1–9 | 10–99 | 100–999 | 1,000+ |
|---|---|---|---|---|---|
| real (held out) | 7.1% | 10.0% | 18.8% | 24.7% | 39.4% |
| `e16_b5` | **22.7%** | **4.6%** | **8.1%** | 22.7% | 41.9% |

- **The engine reproduces the common words.**
- **It writes words seen 1–99 times at less than half the real rate,** and fills the gap with
  made-up ones.

**How well the engine knows each band** (this probe: per-letter surprisal of real held-out
benign path words, nats; lower means better known):

| never | 1–9 | 10–99 | 100–999 | 1,000+ |
|---|---|---|---|---|
| 2.45 | **2.19** | **1.89** | 1.57 | 0.98 |

- **Words seen 1–9 times are nearly as unknown as words never seen.** For an 8-letter word,
  2.19 nats per letter against 0.98 makes it about 16,000 times less likely than a
  well-known word of the same length.
- This is the same pattern the literature reports for letter-by-letter models (§4).

**The size of what is missing** (training word types, benign):

| band | 1–9 | 10–99 | 100–999 | 1,000+ |
|---|---|---|---|---|
| word types | **1,490,862** | **113,644** | 11,332 | 1,043 |
| occurrences | 2.27M | 3.68M | 2.89M | 5.80M |

- **A rough capacity estimate** (an order of magnitude, not a measurement):
  - Making a rare word as familiar as a common one costs about 0.9–1.2 extra nats per
    letter. That is about 10–14 bits per word type.
  - The 1–9 band alone is then about 20 Mbit.
  - Models store about 1 bit per parameter for facts seen about 100 times (§4, Allen-Zhu &
    Li). Words seen 1–9 times get far fewer exposures than that. `e16_cap` has 25M
    parameters for everything.
  - So a letter-by-letter engine of this size cannot hold the tail in its weights.
  - Representations that reuse fragments (pieces) or whole word types need far fewer bits
    per word.

## 2. Your idea, tested

**The idea:** the engine drafts letter by letter; a detector uses the engine's own
uncertainty (entropy) to find suspect words; a reviser rewrites them.

### 2.1 What it needs in order to work

- **A detector that tells made-up words from real rare words.** Real URLs also contain
  rare and new words: 7.1% of benign path words are new even in real data. Flagging those
  would make things worse.
- **A reviser that knows the right word.**
  - If the reviser resamples from the drafter's own knowledge, the result cannot improve.
    Resampling a word from the same model's own conditional distribution leaves that
    model's output distribution exactly as it was (the Gibbs-sampling invariance).
  - If the reviser picks the *most likely* word instead, it removes made-up words, but it
    also removes rare real words in favour of common ones.
  - **It helps only if the reviser knows more than the drafter.**

### 2.2 Test 1, the detector (probe, part D)

The engine's own words in its free files, scored where it wrote them. AUC is the chance
that a made-up word looks more uncertain than a real rare word (0.5 = no better than a
coin):

| field | benign: entropy | benign: surprisal | phishing: entropy | phishing: surprisal |
|---|---|---|---|---|
| name (LBL) | **0.43** | 0.49 | 0.76 | 0.79 |
| subdomain (SUB) | – | – | 0.53 | 0.54 |
| **path (PTH)** | **0.53** | **0.58** | 0.72 | 0.72 |
| query (QRY) | 0.44 | 0.46 | 0.77 | 0.78 |

**Flagging by entropy in benign paths:**

| made-up words caught | real rare words flagged with them | all real words flagged |
|---|---|---|
| 25% | 27% | 14% |
| 50% | **47%** | 27% |
| 75% | 65% | 44% |

- **In benign, where invention is the problem, the detector is at chance.** The engine is
  as unsure writing a rare real word as writing an invention.
- **In phishing it separates better,** but phishing's tell is not invention: it over-uses
  common words (R2). A detector of made-up words does not touch it.

### 2.3 Test 2, the reviser's knowledge (probe, part K)

| question | benign | phishing |
|---|---|---|
| for a slip (a made-up word one edit from a word seen 10+ times), does the engine prefer the corrected URL? | **62%** of 574 (median +1.2 nats) | 67% of 257 |
| for a **real** rare word with a look-alike 10× more common, does the engine prefer the common look-alike? | **69%** of 284 (paths 72%; median +1.9 nats) | 49% of 87 |

- **The engine's "knowledge" is mostly a frequency preference.** It favours the common form
  whether the common form is right (a slip) or wrong (a real rare word).
- **A reviser built on it would destroy rare real words** as readily as it fixes slips.
  That pushes toward over-used common words, which is phishing's failure.

### 2.4 Ceilings already measured (R2, R3)

| change to `e16_b5`'s own URLs | benign path tell closed |
|---|---|
| every slip fixed perfectly, by an oracle that knows the intended word (R2 O1) | **27%** (phishing 4%) |
| every invented word replaced by a real rare word, blind (R2 O2) | 59%, but the host tell more than tripled |
| the engine itself choosing real words in context during writing (R3) | **53%**, but copies of training domains rose 1.4% → 3.9% |

- **R3 is the closest measured version of your idea, with the knowledge it needs supplied.**
  It halved the path tell and still did not pass.
- Its extra knowledge came from a word list, which you ruled out, and it leaked copying
  through names.

### 2.5 What the literature says about revising

- **Two-pass decoding helps when the second pass has information the first lacked.**
  Deliberation networks (Xia et al., 2017) revise translations with the source sentence in
  view. Learned correctors (Welleck et al., 2023) are trained with outside feedback.
- **Self-correction without outside information does not help, and can hurt** (Huang et
  al., ICLR 2024).
- **Revising with a two-sided model of the same data gives worse text than writing left to
  right.** Generating from BERT by Gibbs sampling is more diverse but of slightly lower
  quality (Wang & Cho, 2019).
- **The two-sided conditionals do not even form a consistent model** without a correction
  step (Goyal et al., 2022).
- **Diffusion models that re-mask and revise** catch up with left-to-right models, but do not
  beat them (MDLM 2024: within 14% of their perplexity; ReMDM 2025: "almost matching" them).
  Revision is how they reach left-to-right quality, not how they exceed it.

### 2.6 Verdict on draft-and-revise

- **Not the right solution.** It is a way of sampling, and sampling cannot create knowledge
  the engine lacks.
- In our engine:
  - the detector cannot find the benign fakes (AUC 0.53 in paths);
  - the engine's own preferences would erase rare real words (69%);
  - even perfect spelling repair is worth only 27%.
- **What is right about it:** the problem sits at the uncertain moments where a rare word
  should start. The research use of that signal is to spend more knowledge and computation
  there *while* writing.
  - The Byte Latent Transformer (Pagnoni et al., 2024) starts a new step of its large
    model wherever next-letter entropy is high.
  - It does not revise afterwards.

## 3. The solutions the literature offers, against our evidence

| solution | what it does | evidence on rare words | here | verdict |
|---|---|---|---|---|
| **word pieces (BPE, Sennrich 2016)** | writes paths in learned fragments | words seen 1–99 times: **1.70** bits per character against letters' 2.08; all 7 multilingual languages better than a word list (Mielke & Eisner, 2019) | designed and approved (V2-P1), **never trained**; R1's coverage proxy argued against it | **recommended (§5)** |
| a word list plus spelling for new words (Mielke & Eisner 2019; Kawakami 2017; Luong & Manning 2016) | reuse a known word type, or spell a new one | 1.64 on the 1–99 band; worse than pieces on multilingual text | R3's recommendation; you ruled it out ("copy paste"). R3 leaked copying through names | not better than pieces on multilingual data |
| memory inside the weights (product keys: Lample 2019; Berges et al. 2024) | extra stored parameters | large gains on facts at 1T-token scale | P1/P1b: stored the most repeated strings (site names) first, as memorisation research predicts (Carlini 2023; Lee 2022) | tried; failed K-used |
| retrieval (kNN-LM: Khandelwal 2020; caches: Grave 2017) | looks up real strings while writing | lower perplexity, but **no better open-ended generation** (Wang et al., 2023) | R3's decode-time memory leaked copying | not a route |
| more width or steps | more parameters or exposures | long-tail knowledge needs orders of magnitude more scale (Kandpal 2023; Allen-Zhu 2024) | +77% parameters closed a tenth of the tell (R0) | weak; your last resort |
| frequency-aware training (Gu et al., 2020) | weights rare tokens more in training | helps rare tokens, for models whose tokens are words | needs word counts in training, and has no word units to weight in a letter model | possible add-on later; not first |
| hierarchical byte models (BLT 2024; H-Net 2025) | the model learns word-like chunks itself, without a list | match token models at large scale; no evidence on small rare-word generation | untested; a large rebuild | a future option |
| truncated sampling (Hewitt 2022) | removes low-probability letters | removes made-up words and rare real words alike | decode knobs are closed | closed |
| training against a detector; filtering | pushes outputs toward "real" | language GANs lose to plain training (Caccia 2020) | closed (REVIEW_V2) | closed |
| a pretrained model | outside word knowledge | strongest | closed by you | closed |

**Two notes from neighbouring fields:**
- **Machine-generated domain names (DGAs):** detectors catch letter-built names. The families
  that evade them assemble real dictionary words. That is the same lesson from the
  attacker's side: realism needs real words.
- **Text-to-image models:** models that cannot see letters misspell rendered words; giving
  them letter-level knowledge helps rare words by 30+ points (Liu et al., 2023). The cure is
  knowledge in the representation, not a fix-up pass.

## 4. What the literature says about letter-by-letter models

- **They learn more about words than expected,** even with no word boundaries (Hahn &
  Baroni, 2019). The engine's correct common words show it.
- **They are weakest exactly on rare-but-recurring words.** Mielke & Eisner's frequency bins
  show it (2.08 against 1.70 and 1.64), and so does our probe (§1).
- **Their own authors' verdict:** "pure character-level modeling is not appropriate for
  language, nor required for an open vocabulary" (Mielke & Eisner, 2019).
- **They do not reuse a rare word once used** (Kawakami et al., 2017).
- **Knowledge storage depends on exposure.** About 2 bits per parameter needs about 1,000
  exposures; knowledge seen about 100 times stores at about 1 bit (Allen-Zhu & Li, 2024).
  Facts that are rarer in the training data are learned worse, across model sizes up to
  176B parameters (Kandpal et al., 2023).

## 5. Recommendation: word pieces in the path, query and fragment (corrected in §8: both arms)

**What:** V2's original arm A. The path, query and fragment are written in learned word
pieces. The host and all of `e16_b5`'s machinery stay as they are: the plan, the slash
rules, B5 and the suffix supply. It needs a fresh registration with R1's sizing (the piece
table size is its first decision) and every V2 guard.

**Why it is the best available:**
1. **It attacks the measured cause.** The engine lacks knowledge of rare real words.
   Pieces make every rare word a short combination of known fragments, so it costs far fewer
   bits to learn and to write.
   - Published: 1.70 against 2.08 bits per character on words seen 1–99 times, a fifth
     fewer. Our engine spends 2.7–3.2 bits per letter on that band. A fifth fewer would give
     a typical 8-letter rare word roughly 20–30 times more probability, if the published
     ratio carried over to URLs.
   - Novel words become slightly *less* likely (4.01 against 3.89), so probability moves
     from inventions to real words: the direction we need.
2. **It beats the alternatives on the evidence.**
   - It is as good as a word list on rare words, and better on multilingual text.
   - It needs no outside model.
   - It adds no lookup at generation, so it avoids R3's copying leak.
   - Unlike the memory layer, it does not hand the engine a store that favours repeated
     strings.
3. **It fits your conditions.**
   - No word list chooses the output. The pieces are fragments learned from our own data,
     and the engine still writes freely, piece by piece, coining when it needs to.
   - You approved this design for V2.
4. **It is cheap and informative.** One arm costs about 5–6 GPU-hours, and the per-field
   readings show where it moved the tell.

**How it could fail, and what each failure would teach:**
- **URL vocabularies have a far longer tail than Wikipedia's** (1.49M benign types seen 1–9
  times). The published gain may shrink. R1's coverage figures are the warning sign.
- **Pieces can over-produce common strings** (REVIEW_V2). The copy guards (G7, G-COPY, the
  copier control) catch that.
- **Phishing's tell is over-used common words.** Pieces help calibration in phishing paths
  (R1: 54–66% of phishing path words are held whole), but phishing's larger tell is in the
  host, which arm A leaves alone.

**My odds** (judgement, not registered):
- a cut of at least 30% in the benign path tell: about 55%;
- benign D1-R CLEARED: about 10–15% (R1's estimate was 5%; raised on the published bins);
- phishing CLEARED: about 25%;
- full SUCCESS: about 10%.

**What it would not be:** draft-and-revise, a bigger memory, or a word list. If pieces fall
short, the evidence then points to the word-list family or a much larger engine. Both need
your decision, as they did before.

## 6. Limitations of this investigation

- **The probe reads `e16_b5`'s first free file per class** (1,500 URLs with a path, query or
  fragment) and 1,500 real held-out URLs from the name corpus. It is descriptive, with no
  significance tests.
- **Slips use R1's rule** (one edit from a word seen 10+ times). "Real rare" means training
  count 1–99 in E18's vocabulary.
- **The capacity estimate is rough** (§1).
- **The literature evidence comes from text, not URLs,** and the main comparison
  (Mielke & Eisner) is at WikiText scale.
- **The draft-and-revise verdict rests on three things:** the probe, the ceilings already
  measured and the literature. No reviser was trained. A reviser *with* outside knowledge is
  R3, already measured.

## 7. Sources

- Mielke & Eisner, *Spell Once, Summon Anywhere: A Two-Level Open-Vocabulary Language
  Model*, AAAI 2019 — https://arxiv.org/abs/1804.08205
- Kawakami, Dyer & Blunsom, *Learning to Create and Reuse Words in Open-Vocabulary Neural
  Language Modeling*, ACL 2017 — https://aclanthology.org/P17-1137/
- Hahn & Baroni, *Tabula nearly rasa*, TACL 2019 — https://aclanthology.org/Q19-1033/
- Xia et al., *Deliberation Networks*, NeurIPS 2017 —
  https://proceedings.neurips.cc/paper/2017/hash/c6036a69be21cb660499b75718a3ef24-Abstract.html
- Welleck et al., *Generating Sequences by Learning to Self-Correct*, ICLR 2023 —
  https://arxiv.org/abs/2211.00053
- Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet*, ICLR 2024 —
  https://arxiv.org/abs/2310.01798
- Wang & Cho, *BERT has a Mouth, and It Must Speak*, 2019 — https://arxiv.org/abs/1902.04094
- Goyal, Dyer & Berg-Kirkpatrick, *Exposing the Implicit Energy Networks behind Masked
  Language Models via Metropolis–Hastings*, ICLR 2022 — https://arxiv.org/abs/2106.02736
- Sahoo et al., *Simple and Effective Masked Diffusion Language Models*, 2024 —
  https://arxiv.org/abs/2406.07524
- Wang et al., *Remasking Discrete Diffusion Models with Inference-Time Scaling*, NeurIPS
  2025 — https://arxiv.org/abs/2503.00307
- Pagnoni et al., *Byte Latent Transformer: Patches Scale Better Than Tokens*, 2024 —
  https://arxiv.org/abs/2412.09871
- Hwang, Wang & Gu, *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling*, 2025 —
  https://arxiv.org/abs/2507.07955
- Carlini et al., *Quantifying Memorization Across Neural Language Models*, ICLR 2023 —
  https://arxiv.org/abs/2202.07646
- Lee et al., *Deduplicating Training Data Makes Language Models Better*, ACL 2022 —
  https://arxiv.org/abs/2107.06499
- Kandpal et al., *Large Language Models Struggle to Learn Long-Tail Knowledge*, ICML 2023 —
  https://arxiv.org/abs/2211.08411
- Allen-Zhu & Li, *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws*,
  ICLR 2025 — https://arxiv.org/abs/2404.05405
- Berges et al., *Memory Layers at Scale*, ICML 2025 — https://arxiv.org/abs/2412.09764
- Wang et al., *KNN-LM Does Not Improve Open-ended Text Generation*, EMNLP 2023 —
  https://arxiv.org/abs/2305.14625
- Hewitt, Manning & Liang, *Truncation Sampling as Language Model Desmoothing*, EMNLP
  Findings 2022 — https://arxiv.org/abs/2210.15191
- Caccia et al., *Language GANs Falling Short*, ICLR 2020 — https://arxiv.org/abs/1811.02549
- Gu et al., *Token-level Adaptive Training for Neural Machine Translation*, EMNLP 2020 —
  https://arxiv.org/abs/2010.04380
- Liu et al., *Character-Aware Models Improve Visual Text Rendering*, ACL 2023 —
  https://arxiv.org/abs/2212.10562
- He et al., *Exposure Bias versus Self-Recovery*, EMNLP 2021 —
  https://arxiv.org/abs/1905.10617
- Sennrich, Haddow & Birch, *Neural Machine Translation of Rare Words with Subword Units*,
  ACL 2016 — https://arxiv.org/abs/1508.07909
- On dictionary-based domain generation algorithms: Splunk, *Threat Hunting for
  Dictionary-DGA with PEAK* —
  https://www.splunk.com/en_us/blog/security/threat-hunting-for-dictionary-dga-with-peak

## 8. Correction (2026-10-13, at the owner's question): the host belongs in the test

**The owner's question, verbatim:**

> If we leave the domain as letter-by-letter generation, how exactly does this solve the
> phishing host tell? Are we just choosing to ignore the highest tell in the phishing class
> to protect domain creativity?

**The answer: arm A alone would not touch it.**
- Recommending arm A alone was a mistake. It repeats the error I admitted in REVIEW_V2:
  leaving the host out without testing the reason.
- It also went against the owner's decision of 2026-10-09, P1 = arms A and B.

**The phishing host tell is the class's largest:**
- 0.147 on the host against 0.100 on the path, query and fragment (R0). The honest judge
  reads it at 0.102 (R2).
- Phishing subdomains (0.118) are the class's largest single field (R1).

**Its nature is the one pieces address.**
- **Invention, plus a shortfall of rare real words.**
  - R1: never-seen words rise from 22.7% to 30.8% in phishing names and from 25.1% to
    33.5% in subdomains. Words beyond the 32,000 commonest fall from 4.1% to 1.5% and from
    5.1% to 2.4%.
  - This probe: never-seen words in real against generated phishing names, 21.9% against
    39.8%; subdomains, 29.8% against 48.2%.
- **Pieces fit phishing hosts well.** One piece holds 51–59% of phishing host words whole
  (R1-4), against 34–38% of benign host words and 32–42% of benign path words.
  - *Corrected 2026-10-13:* the first version said "better than any other field". Phishing
    paths are held better still (54–66%).
- **What failed in hosts before was forcing whole real words into them.** R2's blind oracle
  doubled the phishing host tell; R3 raised copies of training domains from 1.4% to 3.9%.
  Pieces do not force words.
- **In benign names, real never-seen words are often two real words joined (34%).** The
  engine joins less and coins more (R1-3). Pieces make that kind of composition natural.

**Creativity is not a reason to leave the host out.**
- The engine's names are already *more* invented than real names. That excess is the tell.
- Creativity is measured (new names at the real rate, host word diversity, new registrable
  domains), not protected by exclusion.

**The real hazard is copying, and it lives in how a piece table is learned**
(`v2_piece_hazard.py`, e75b324; output `data/v2/dr_probe/piece_hazard.json`):

| table (32,000 pieces) | pieces that are a whole training string | strings drawn 50+ times that became one piece | pieces of 20+ characters |
|---|---|---|---|
| host, as R1 learned it (drawn as training draws: a site's many pages count many times) | 6,634 (20.7%) | 184 of 189 | 191 |
| host, each distinct string counted once | 4,293 (13.4%) | 103 of 189 | 84 |
| path/query/fragment, as R1 learned it | 2,495 (7.8%) | 208 of 210 | 489 |
| path/query/fragment, each distinct string counted once | 2,018 (6.3%) | 104 of 210 | 365 |

- **R1's host table holds verbatim phishing hosts as single pieces,** for example
  `turkiyecumhuriyetiziraatbankasi` and `metamask-docs-45ta0ykbu-consensys-ddffed67`.
  - An engine writing them would regenerate real phishing domains.
  - That is P1b's memorisation, inside the tokenizer.
- **The path table has the same problem:** 208 of 210 heavily drawn strings, and junk pieces
  of 100+ characters. Arm A needs the fix too.
- **Counting each string once helps, but is not enough.** Campaign families that share a stem
  still merge (`case10009…`).

**The revised recommendation: both arms, as the owner decided on 2026-10-09, with a piece
table built against memorisation.**
- **Arm A:** pieces in the path, query and fragment. **Arm B:** arm A plus pieces in the
  host. Data, recipe and start are matched.
- **The piece table:**
  - each site counted once, not each page;
  - a cap on piece length (real words are short);
  - a check, before any training, that no piece is a whole training host or path string.

  The exact rules are the registration's first decisions, measured on the laptop.
- **The host guards:**
  - copies of registrable domains against the natural rate (phishing's is high: 63.7%;
    `e16_b5` 56.8%);
  - regeneration of live phishing domains, with a measure that works (R1 L7: the old
    snapshot matched shared platforms);
  - host word diversity;
  - new names at the real rate.
- **The pass mark must read the host too,** through the honest judge on the host or on the
  whole URL. Otherwise arm B's purpose goes ungraded.
- **Cost:** about 10–11 GPU-hours for both arms (REVIEW_V2), plus scoring on the laptop.
- **My odds (judgement, not registered):**
  - in arm B, a cut of at least 30% in the phishing host tell: about 40%;
  - phishing host CLEARED: about 15%;
  - benign as in §5.
