"""
uie/corpus/pack_kaggle.py

Build the upload bundle for GPU training on Kaggle.

Why a subset rather than the whole corpus
-----------------------------------------
The full corpus is ~1 GB and 6.85M URLs. A character model over a ~140-token
vocabulary saturates well before that, and upload time is real time. So benign
has always been subsampled. The question this module answers is not WHETHER to
subsample but WHICH subset to keep.

Why the subset is capped per name, not drawn at random  (E2.1)
--------------------------------------------------------------
E2 drew 1.2M benign URLs uniformly at random. Measured consequences:

    benign corpus     6,496,447 URLs over    99,303 registrable labels (65.4/label)
    E2 bundle         1,200,000 URLs over    55,004 labels
    top-1000 labels   carried 38.4% of the entire label-field gradient

A uniform draw preserves the corpus's head-heavy shape, so the model spent most
of its label-field learning on a few thousand labels and saw the long tail —
where the actual morphology of Internet naming lives — only in passing. The
per-field diagnostic showed exactly what that produces:

    LBL  benign   train NLL 1.201   val NLL 2.346   gap +1.144  <- worst in model
    PTH  benign   train NLL 1.727   val NLL 1.987   gap +0.259

The label field is the most OVERFIT field in the model. The validation set is
99.79% label-disjoint from training, so that memorization buys nothing at
evaluation time: the model learned WHICH labels exist instead of HOW labels are
built, and when asked to invent one it had no structure to fall back on. That is
the syllabic-mush failure E3 caught as real-vs-fake AUC 0.868/0.828.

Capping URLs per name fixes the sampling, not the corpus:

       K      URLs kept    labels   top-1000 share
      32      1,017,550    79,473        3.1%
   uncapped   5,276,148    79,473       38.4%
   E2 (rand)  1,200,000    55,004       38.4%

Every training label, 44% more than E2, in FEWER URLs than E2 uploaded. Nothing
is lost relative to the baseline — 77% of benign URLs were already being
discarded; this just chooses a better million.

Capping by label string does NOT remove morphemes. `news`, `shop`, `secure` and
`daily` recur across thousands of distinct labels, so their frequency barely
moves. What stops is `wordpress` THE STRING being seen 200,000 times. That is
precisely the trade the diagnosis calls for.

Why the two classes cap on different fields
-------------------------------------------
The invented name does not live in the same place in both classes:

    vercel    26,773 phishing URLs across 17,494 distinct subdomain prefixes
    webflow   21,548 phishing URLs across 18,729 distinct subdomain prefixes

Benign URLs name the site in the REGISTRABLE LABEL, and their head is one site
with thousands of pages — redundancy worth capping. Phishing URLs are mostly on
shared hosting, where the label is the PLATFORM and the attacker's invented name
is the SUBDOMAIN; the phishing head is 17,494 different names, not one name
repeated. Capping phishing by label would delete the very diversity we want.

So: one principle — cap by the unit that carries the invented name — applied to
whichever field that is. Benign caps by label, phishing by full host. Phishing
holds 1.5 URLs/host, so its cap removes only path-level duplication (85% kept).

Validation is NEVER capped
--------------------------
The benchmark must not move when the training set does. Val is drawn exactly as
E2 drew it, so E2.1's numbers are comparable to E2's; a metric that shifts with
the thing it is measuring cannot adjudicate a pre-registered criterion.

The split is respected, not re-drawn
------------------------------------
Rows come from the FROZEN `train` and `val` partitions. Re-splitting here would
silently produce a different, unverifiable division — the exact failure
`load_split`'s hash check exists to prevent — and would let val domains leak.

What ends up in the bundle
--------------------------
The `uie` package travels with the data, so the notebook needs no network access
and no pip install from an unpublished repository. A Bloom filter over EVERY
training URL travels with it too — see `known_urls.bloom.gz` below.
"""
from __future__ import annotations

import glob
import gzip
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from uie.bloom import BloomFilter
from uie.factorization import parse
from uie.gate import normalize_for_dedup
from uie.splits import load_split

#: Max benign URLs kept per registrable label. 32 covers all 79,473 training
#: labels in ~1.02M URLs — fewer than E2 uploaded — and drops the top-1000
#: labels' share of the label-field gradient from 38.4% to 3.1%.
BENIGN_LABEL_CAP = 32

#: Phishing is NOT capped.  (E2.1 tried `PHISH_HOST_CAP = 4` and it failed.)
#:
#: The cap exists to flatten a skewed frequency distribution over the field that
#: carries the invented name. Phishing's was ALREADY FLAT — 1.5 URLs per host
#: across 206,306 hosts — so there was no skew to remove. What the cap did do was
#: cut the scarce class 303,176 -> 258,851, which at unchanged steps and 50/50
#: class balancing raised phishing from ~8.4 epochs to ~9.9. Its path field
#: promptly over-trained (train NLL 1.625 -> 1.360 while val rose 2.229 -> 2.285)
#: and every phishing component got HARDER to distinguish from real, by +0.027 to
#: +0.029 AUC.
#:
#: The lesson is about where an intervention applies, not about capping: applied
#: to benign, where the skew was real, the same mechanism won −0.094 on the label
#: tell. Applied where the distribution was already flat, it was pure data loss.
PHISH_HOST_CAP: Optional[int] = None

#: Validation caps: unchanged from E2 so the benchmark stays comparable.
BENIGN_VAL_CAP = 60_000
PHISH_VAL_CAP = 25_000

#: False-positive rate for the novelty Bloom filter. Errors are one-directional
#: (a novel URL wrongly called memorized), so this UNDER-states novelty by at
#: most 0.1% and can never overstate it.
BLOOM_ERROR_RATE = 0.001


def _host_key(url: str) -> str:
    parts = parse(url)
    return ".".join(tuple(parts.subdomains) +
                    (parts.registrable_label, parts.public_suffix))


def _label_key(url: str) -> str:
    return parse(url).registrable_label


class _Capped:
    """Keeps at most `cap` rows per key, sampled uniformly in ONE pass.

    Reservoir sampling (Algorithm R) rather than "first K seen": the shards are
    ordered by crawl and by SURT, so taking the first K would bias every label
    toward whichever crawl and whichever path prefix happens to sort first.
    Memory is bounded at cap x n_keys regardless of corpus size.
    """

    def __init__(self, cap: Optional[int], key: Callable[[str], str],
                 rng: random.Random) -> None:
        #: `None` keeps everything. Not the same as a very large cap: it states
        #: in the manifest that this class was deliberately left uncapped,
        #: rather than capped at a number that happened never to bind.
        self.cap = cap
        self.key = key
        self.rng = rng
        self.kept: Dict[str, List[dict]] = {}
        self.seen: Dict[str, int] = {}

    def offer(self, row: dict) -> None:
        try:
            key = self.key(row["url"])
        except Exception:
            key = ""
        if not key:
            key = row["url"]                 # IP hosts and unparseable oddities
        count = self.seen.get(key, 0) + 1
        self.seen[key] = count
        bucket = self.kept.setdefault(key, [])
        if self.cap is None or count <= self.cap:
            bucket.append(row)
            return
        # Replace a uniformly chosen slot with probability cap/count, which
        # leaves every one of the `count` rows equally likely to survive.
        index = self.rng.randrange(count)
        if index < self.cap:
            bucket[index] = row

    def rows(self) -> List[dict]:
        return [row for bucket in self.kept.values() for row in bucket]

    def stats(self) -> Dict[str, int]:
        return {"keys": len(self.kept), "kept": sum(map(len, self.kept.values())),
                "seen": sum(self.seen.values())}


def _write(rows: List[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path.stat().st_size


def _iter_corpus(corpus: Path):
    """Yield (url, label, language) for the whole corpus, streaming."""
    for path in sorted(glob.glob(str(corpus / "shards" / "*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                # Common Crawl reports several languages per page; the first is
                # the dominant one and is what a control token should carry.
                lang = str(row.get("languages") or "und").split(",")[0].strip()
                yield row["url"], "benign", lang or "und"

    phishing = corpus / "phishing.jsonl"
    if phishing.exists():
        with open(phishing, encoding="utf-8") as handle:
            for line in handle:
                # The feed carries no language label. "und" is honest; guessing
                # would fabricate a control signal that does not exist.
                yield json.loads(line)["url"], "phishing", "und"


def build(corpus: Path, out: Path, *, seed: int = 20260803,
          benign_label_cap: int = BENIGN_LABEL_CAP,
          phish_host_cap: int = PHISH_HOST_CAP) -> dict:
    split = load_split(corpus / "split_random.json.gz")

    # Independent streams per consumer. E2 drew train and val from ONE Random,
    # so any change to train sampling silently reshuffled val too — which would
    # have moved the benchmark underneath the experiment it is judging.
    rng_benign = random.Random(seed)
    rng_phish = random.Random(seed ^ 0x9E3779B9)
    rng_val = random.Random(seed ^ 0x5A5A5A5A)

    capped = {
        "benign": _Capped(benign_label_cap, _label_key, rng_benign),
        "phishing": _Capped(phish_host_cap, _host_key, rng_phish),
    }
    val_pool: Dict[str, List[dict]] = {}

    # Sized for the full training partition, not the bundle: novelty is checked
    # against every URL the model could have seen, including those the cap drops.
    known = BloomFilter(capacity=6_000_000, error_rate=BLOOM_ERROR_RATE)

    for url, label, lang in _iter_corpus(corpus):
        part = split.partition_of(url)
        if part == "train":
            known.add(normalize_for_dedup(url))
            capped[label].offer({"url": url, "label": label, "lang": lang})
        elif part == "val":
            val_pool.setdefault(label, []).append(
                {"url": url, "label": label, "lang": lang})

    def sample(rows: List[dict], cap: Optional[int]) -> List[dict]:
        if cap is None or len(rows) <= cap:
            return rows
        return rng_val.sample(rows, cap)

    train_rows = capped["benign"].rows() + capped["phishing"].rows()
    val_rows = (sample(val_pool.get("benign", []), BENIGN_VAL_CAP) +
                sample(val_pool.get("phishing", []), PHISH_VAL_CAP))
    rng_benign.shuffle(train_rows)
    rng_val.shuffle(val_rows)

    out.mkdir(parents=True, exist_ok=True)
    train_bytes = _write(train_rows, out / "train.jsonl.gz")
    val_bytes = _write(val_rows, out / "val.jsonl.gz")
    bloom_bytes = known.save(out / "known_urls.bloom.gz")

    # Ship the package alongside the data: one dataset, no network needed.
    package_src = Path(__file__).resolve().parent.parent
    package_dst = out / "uie"
    if package_dst.exists():
        shutil.rmtree(package_dst)
    shutil.copytree(package_src, package_dst,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    def tally(rows: List[dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for row in rows:
            counts[row["label"]] = counts.get(row["label"], 0) + 1
        return counts

    manifest = {
        "seed": seed,
        "split_hash": split.content_hash(),
        "train": {"n": len(train_rows), "by_class": tally(train_rows),
                  "bytes": train_bytes},
        "val": {"n": len(val_rows), "by_class": tally(val_rows),
                "bytes": val_bytes},
        "sampling": {
            "benign": {"cap_field": "registrable_label",
                       "cap": benign_label_cap, **capped["benign"].stats()},
            "phishing": {"cap_field": "host", "cap": phish_host_cap,
                         **capped["phishing"].stats()},
            "val": {"capped": False, "benign_cap": BENIGN_VAL_CAP,
                    "phishing_cap": PHISH_VAL_CAP},
        },
        "known_urls_bloom": {
            "n_added": known.n_added, "bytes": bloom_bytes,
            "expected_error_rate": round(known.expected_error_rate, 6),
        },
    }
    (out / "bundle.manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    # Zip in the same step, never separately. Kaggle's uploader takes files, not
    # folders, so a zip is required — and a zip built at a different time than
    # the bundle is how a STALE COPY OF THE PACKAGE gets uploaded, which fails
    # only once it is already running on the GPU.
    archive = out.parent / "uie-corpus.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out.rglob("*")):
            if path.is_file() and "__pycache__" not in str(path):
                zf.write(path, path.relative_to(out))
    manifest["archive"] = str(archive)
    manifest["archive_bytes"] = archive.stat().st_size
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the Kaggle upload bundle.")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--out", type=Path, default=Path("data/kaggle_bundle"))
    ap.add_argument("--benign-label-cap", type=int, default=BENIGN_LABEL_CAP)
    ap.add_argument("--phish-host-cap", type=int, default=PHISH_HOST_CAP)
    args = ap.parse_args()

    man = build(args.corpus, args.out, benign_label_cap=args.benign_label_cap,
                phish_host_cap=args.phish_host_cap)
    for cls, info in man["sampling"].items():
        if "keys" in info:
            print(f"{cls:<9} {info['kept']:>9,} URLs over {info['keys']:>7,} "
                  f"{info['cap_field']}s  (cap {info['cap']}, "
                  f"from {info['seen']:,})")
    print(f"train {man['train']['n']:>9,}  {man['train']['by_class']}")
    print(f"val   {man['val']['n']:>9,}  {man['val']['by_class']}")
    print(f"bloom {man['known_urls_bloom']['n_added']:>9,} URLs  "
          f"({man['known_urls_bloom']['bytes']/1048576:.1f} MB, "
          f"fp={man['known_urls_bloom']['expected_error_rate']:.4%})")
    print(f"\nUPLOAD THIS FILE:  {man['archive']}  "
          f"({man['archive_bytes']/1048576:.1f} MB)")
