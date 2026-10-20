"""
research/eval_pages/build_eval_sets.py

Draws the two human-evaluation sets and fills the two page templates.

    python research/eval_pages/build_eval_sets.py

Sources (E18's, research/e18/strong_disc.py):
- real: the gate's whole-URL population, the `mixed` validation pool of data/bundle2, one URL
  per host, normalised by FIELDS["FULL"]; never used in training.
- generated: e16_b5's free-mode files, data/e17/decode/gen/free_e16_b5_{label}_g{g}.json.

The two sets:
- laptop (100 URLs, 10 participants x 5 pairs): 25 real + 25 generated per class, random
  draws, 5 from each generation file. Each participant's block holds 5 real and 5 generated
  (odd blocks 3+3 benign and 2+2 phishing, even blocks 2+2 and 3+3), shown as 5 same-class
  pairs: 3 real-vs-generated, 1 both real, 1 both generated, with left/right randomised.
  The rater picks the one that "feels real".
- remote (10 URLs, one person): 5 generated (3 benign, 2 phishing), each drawn with a real URL
  of the same class, the same path and query presence, and a length within 20%; shown as 5
  pairs like the laptop's (benign: both real, both generated, one mixed; phishing: two mixed).
Both pages are filled from one template, human_eval_pairs.html.

Rules for every URL: no email address in it; no host used twice anywhere in either set; no
URL shared between the sets. IDs are assigned in a shuffled order, so they say nothing about
the source.

Outputs, all under data/eval_pages/ (git-ignored: the pages hold real phishing URLs):
- human_eval_100_laptop.html, human_eval_10_remote.html: the filled pages;
- keys/laptop_key.json, keys/remote_key.json: the answer keys (do not open before grading);
- MANIFEST.json: seeds, counts and the sha256 of every file.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "research" / "cr1", ROOT / "research" / "native"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

BUNDLE = ROOT / "data" / "bundle2"
GEN = ROOT / "data" / "e17" / "decode" / "gen" / "free_e16_b5_{label}_g{g}.json"
OUT = ROOT / "data" / "eval_pages"
LABELS = ("benign", "phishing")
G = 5
SEED = 20261015                     # registered here, before any draw
PER_CELL = 25                       # laptop: real/generated x benign/phishing
REMOTE_GEN = {"benign": 3, "phishing": 2}
LENGTH_TOL = 0.20
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SET_BLOCK = re.compile(r'(<script type="application/json" id="uie-set">\n)(.*?)(\n</script>)', re.S)


def host_of(u: str) -> str:
    return re.split(r"[/?#]", u, maxsplit=1)[0].lower()


def shape(u: str) -> tuple:
    """(has a path body, has a query, length): what the remote pairs are matched on."""
    rest = u[len(host_of(u)):]
    path = re.split(r"[?#]", rest, maxsplit=1)[0]
    query = rest.split("?", 1)[1].split("#", 1)[0] if "?" in rest else ""
    return bool(path.strip("/")), bool(query), len(u)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def real_pool(label: str) -> list:
    from cr1_score import pools
    from uie.evaluation import FIELDS, _one_per_host
    pop = pools(BUNDLE, label)["mixed"]
    out = [x for x in (FIELDS["FULL"](u) for u in _one_per_host(pop, SEED + LABELS.index(label))) if x]
    random.Random(SEED + 10 + LABELS.index(label)).shuffle(out)
    return out


def gen_pool(label: str) -> list:
    """[(g, index, url)] over the five free files, each file shuffled with its own seed."""
    from uie.evaluation import FIELDS
    out = []
    for g in range(G):
        urls = json.loads(Path(str(GEN).format(label=label, g=g)).read_text(encoding="utf-8"))
        rows = [(g, i, FIELDS["FULL"](u)) for i, u in enumerate(urls)]
        rows = [r for r in rows if r[2]]
        random.Random(SEED + 100 + 10 * LABELS.index(label) + g).shuffle(rows)
        out.append(rows)
    return out


class Picker:
    """Draws URLs in pool order, skipping any that break the rules."""

    def __init__(self):
        self.hosts, self.urls, self.skipped = set(), set(), {"email": 0, "host_or_url_reused": 0}

    def ok(self, u: str) -> bool:
        if EMAIL.search(u):
            self.skipped["email"] += 1
            return False
        if u in self.urls or host_of(u) in self.hosts:
            self.skipped["host_or_url_reused"] += 1
            return False
        return True

    def take(self, u: str) -> None:
        self.urls.add(u)
        self.hosts.add(host_of(u))


def draw_gen(pick: Picker, rows_by_g: list, n: int, cursor: dict, label: str) -> list:
    """n generated URLs, spread evenly over the generation files (round robin)."""
    out, g = [], 0
    while len(out) < n:
        rows = rows_by_g[g % G]
        k = cursor.setdefault((label, g % G), 0)
        while k < len(rows) and not pick.ok(rows[k][2]):
            k += 1
        if k >= len(rows):
            raise SystemExit(f"STOP: generation file g{g % G} ({label}) ran out")
        out.append({"source": "e16_b5", "class": label, "url": rows[k][2], "file": f"g{rows[k][0]}", "index": rows[k][1]})
        pick.take(rows[k][2])
        cursor[(label, g % G)] = k + 1
        g += 1
    return out


def draw_real(pick: Picker, pool: list, n: int, cursor: dict, label: str, want=None) -> list:
    out = []
    k = cursor.setdefault(("real", label), 0)
    skipped = []
    while len(out) < n:
        if k >= len(pool):
            raise SystemExit(f"STOP: real {label} pool ran out")
        u = pool[k]
        k += 1
        if not pick.ok(u):
            continue
        if want is not None:
            hp, hq, ln = shape(u)
            if (hp, hq) != want[:2] or abs(ln - want[2]) > LENGTH_TOL * want[2]:
                skipped.append(u)
                continue
        out.append({"source": "real", "class": label, "url": u, "pool_rank": k - 1})
        pick.take(u)
    if want is None:
        cursor[("real", label)] = k
    return out


def fill(template: Path, set_obj: dict) -> str:
    html = template.read_text(encoding="utf-8")
    m = SET_BLOCK.search(html)
    if not m:
        raise SystemExit(f"STOP: no set block in {template.name}")
    body = json.dumps(set_obj, indent=1, ensure_ascii=False)
    if "</script" in body.lower():
        raise SystemExit("STOP: a URL contains '</script'")
    return html[:m.start(2)] + body + html[m.end(2):]


def finish_set(items: list, prefix: str, width: int, seed: int) -> tuple:
    """Assigns shuffled IDs; returns (page items, key)."""
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    ids = {}
    for n, i in enumerate(order, start=1):
        ids[i] = f"{prefix}{n:0{width}d}"
    page, key = [], {}
    for i, it in enumerate(items):
        row = {"id": ids[i], "class": it["class"], "url": it["url"]}
        if "block" in it:
            row["block"] = it["block"]
        page.append(row)
        key[ids[i]] = {k: v for k, v in it.items() if k != "url"} | {"url": it["url"]}
    page.sort(key=lambda r: r["id"])
    return page, dict(sorted(key.items()))


def pair_block(block_items: list, rng: random.Random) -> list:
    """5 same-class pairs from one block: the class with 3+3 gives real-real, generated-generated
    and one mixed pair; the class with 2+2 gives two mixed pairs."""
    pairs = []
    for lab in LABELS:
        r = [x for x in block_items if x["class"] == lab and x["source"] == "real"]
        g = [x for x in block_items if x["class"] == lab and x["source"] == "e16_b5"]
        if len(r) == 3:
            plan = [(r[0], r[1]), (g[0], g[1]), (r[2], g[2])]
        else:
            plan = [(r[0], g[0]), (r[1], g[1])]
        for x, y in plan:
            if rng.random() < 0.5:
                x, y = y, x
            pairs.append({"class": lab, "block": x["block"], "a": x, "b": y})
    return pairs


def finish_pairs(pairs: list, seed: int, prefix: str = "Q", width: int = 3) -> tuple:
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    page, key = [], {}
    for n, i in enumerate(order, start=1):
        pr, pid = pairs[i], f"{prefix}{n:0{width}d}"
        page.append({"id": pid, "block": pr["block"], "class": pr["class"], "a": pr["a"]["url"], "b": pr["b"]["url"]})
        kinds = sorted(z["source"] for z in (pr["a"], pr["b"]))
        key[pid] = {"type": {"e16_b5,real": "real-vs-generated", "real,real": "both-real",
                             "e16_b5,e16_b5": "both-generated"}[",".join(kinds)],
                    "a": pr["a"], "b": pr["b"]}
    page.sort(key=lambda r: r["id"])
    return page, dict(sorted(key.items()))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pick = Picker()
    cursor = {}
    real = {lab: real_pool(lab) for lab in LABELS}
    gen = {lab: gen_pool(lab) for lab in LABELS}
    for lab in LABELS:
        print(f"{lab}: real pool {len(real[lab]):,} (one per host); generated {sum(len(r) for r in gen[lab]):,} in {G} files")

    # ---- remote: 5 matched pairs ----
    remote = []
    for lab in LABELS:
        for g_item in draw_gen(pick, gen[lab], REMOTE_GEN[lab], cursor, lab):
            match = draw_real(pick, real[lab], 1, {}, lab, want=shape(g_item["url"]))[0]
            pair = len(remote) // 2 + 1
            remote += [g_item | {"pair": pair}, match | {"pair": pair}]

    # ---- laptop: 25 per cell, then blocks ----
    cells = {(lab, src): [] for lab in LABELS for src in ("real", "e16_b5")}
    for lab in LABELS:
        cells[(lab, "e16_b5")] = draw_gen(pick, gen[lab], PER_CELL, cursor, lab)
        cells[(lab, "real")] = draw_real(pick, real[lab], PER_CELL, cursor, lab)
    for c in cells.values():
        random.Random(SEED + 7).shuffle(c)
    laptop = []
    for b in range(1, 11):
        nb, np_ = (3, 2) if b % 2 else (2, 3)
        for (lab, n) in (("benign", nb), ("phishing", np_)):
            for src in ("real", "e16_b5"):
                for _ in range(n):
                    laptop.append(cells[(lab, src)].pop() | {"block": b})
    assert all(not c for c in cells.values()), "every drawn URL is placed"

    # ---- checks ----
    all_urls = [x["url"] for x in remote + laptop]
    assert len(all_urls) == 110 and len(set(all_urls)) == 110, "110 distinct URLs"
    assert len({host_of(u) for u in all_urls}) == 110, "110 distinct hosts"
    for b in range(1, 11):
        blk = [x for x in laptop if x["block"] == b]
        assert len(blk) == 10 and sum(x["source"] == "real" for x in blk) == 5
    for lab in LABELS:
        for src in ("real", "e16_b5"):
            assert sum(1 for x in laptop if x["class"] == lab and x["source"] == src) == PER_CELL

    # ---- write ----
    (OUT / "keys").mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SEED, "sources": {"real": "data/bundle2 mixed validation pool, one per host, FIELDS['FULL']",
                                          "generated": str(GEN.relative_to(ROOT))},
                "skipped_while_drawing": pick.skipped, "files": {}}
    rng_pairs = random.Random(SEED + 3)
    laptop_pairs = [pr for b in range(1, 11) for pr in pair_block([x for x in laptop if x["block"] == b], rng_pairs)]
    assert len(laptop_pairs) == 50
    # Mixed pairs: the real URL goes on side A in exactly half of them, so a side preference earns nothing.
    mixed = [pr for pr in laptop_pairs if pr["a"]["source"] != pr["b"]["source"]]
    random.Random(SEED + 4).shuffle(mixed)
    for i, pr in enumerate(mixed):
        real_first = i < len(mixed) // 2
        if (pr["a"]["source"] == "real") != real_first:
            pr["a"], pr["b"] = pr["b"], pr["a"]
    # The remote person's 10 URLs as 5 pairs (the laptop set above is untouched by this).
    remote_pairs = pair_block([x | {"block": 1} for x in remote], random.Random(SEED + 5))
    assert len(remote_pairs) == 5
    for name, items, pairs, prefix, width, study, out_name, participants, set_seed in (
            ("laptop", laptop, laptop_pairs, "Q", 3, "uie-human-100", "human_eval_100_laptop.html", 10, SEED + 1),
            ("remote", remote, remote_pairs, "R", 2, "uie-human-10", "human_eval_10_remote.html", 1, SEED + 2)):
        page_items, key = finish_pairs(pairs, set_seed, prefix, width)
        canon = json.dumps(page_items, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        set_sha = sha256_bytes(canon)
        set_id = f"{name}-{SEED}"
        set_obj = {"study": study, "format": 1, "set_id": set_id, "set_sha256": set_sha, "placeholder": False,
                   "participants": participants, "items": page_items}
        html = fill(HERE / "human_eval_pairs.html", set_obj)
        page_path = OUT / out_name
        page_path.write_text(html, encoding="utf-8", newline="\n")
        key_path = OUT / "keys" / f"{name}_key.json"
        key_path.write_text(json.dumps({"set_id": set_id, "set_sha256": set_sha, "items": key}, indent=1, ensure_ascii=False),
                            encoding="utf-8")
        manifest["files"][name] = {
            "set_id": set_id, "set_sha256": set_sha, "n": len(page_items),
            "page": str(page_path.relative_to(ROOT)), "page_sha256": sha256_bytes(page_path.read_bytes()),
            "key": str(key_path.relative_to(ROOT)), "key_sha256": sha256_bytes(key_path.read_bytes()),
            "classes": {lab: sum(1 for x in items if x["class"] == lab) for lab in LABELS},
            "sources": {s: sum(1 for x in items if x["source"] == s) for s in ("real", "e16_b5")}}
        what = f"{len(page_items)} pairs ({2 * len(page_items)} URLs)"
        print(f"{name}: {what} -> {page_path.relative_to(ROOT)}; set sha256 {set_sha[:12]}")
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("skipped while drawing:", pick.skipped)
    print("wrote", (OUT / "MANIFEST.json").relative_to(ROOT))


if __name__ == "__main__":
    main()
