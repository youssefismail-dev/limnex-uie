"""
research/d2/make_release.py — D2 step 5: build the release candidate (research/d2/D2_PLAN.md).

    python research/d2/make_release.py --repo ../limnex-uie --files ../limnex-uie-release-files

1. The public repository (decision C): a new folder holding ONE commit whose tree is exactly this
   repository's HEAD tree, so the audited files byte for byte, authored by the owner with GitHub's
   no-reply address and no AI co-author line (decision F). It has no remote: going public is the
   owner's step (decision E), following LAUNCH_STEPS.md.
2. The release files (decision D), under --files: attach/ holds the model and the known-URL filter
   under their published names, checked against the fingerprints uie.load enforces, with
   SHA256SUMS.txt; RELEASE_NOTES.md is the text for the release page, LAUNCH_STEPS.md the owner's
   launch-day steps.

Refuses a dirty tree, an existing target folder, or a file that is not the published one. The
history of this repository never leaves it: the new folder fetches only the new commit's objects.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uie import __version__  # noqa: E402
from uie.api import (PUBLISHED_KNOWN_URLS_SHA256, PUBLISHED_WEIGHTS_SHA256,  # noqa: E402
                     _content_sha256, _sha256)

OWNER_NAME = "youssefismail-dev"
OWNER_EMAIL = "255600562+youssefismail-dev@users.noreply.github.com"   # GitHub's no-reply address
PUBLIC_REPO = "https://github.com/youssefismail-dev/limnex-uie"
TAG = f"v{__version__}"
TEMP_REF = "refs/release/candidate"
MODEL, FILTER = "limnex-uie-e16_b5.pt", "limnex-uie-known-urls.bloom.gz"
MESSAGE = f"""Limnex URL Intelligence Engine {__version__}

The first public release of Limnex UIE: the code, the documents and the research records of
the e16_b5 engine. The model file and the known-URL filter are attached to the {TAG} release.
Read the README's Responsible Use and No Liability sections before any use.
"""


def git(*args, cwd=ROOT, env=None) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def build_repo(repo: Path) -> dict:
    if git("status", "--porcelain"):
        raise SystemExit("the working tree is not clean: commit first, so the release is a recorded state")
    tree, head = git("rev-parse", "HEAD^{tree}"), git("rev-parse", "HEAD")
    who = {"GIT_AUTHOR_NAME": OWNER_NAME, "GIT_AUTHOR_EMAIL": OWNER_EMAIL,
           "GIT_COMMITTER_NAME": OWNER_NAME, "GIT_COMMITTER_EMAIL": OWNER_EMAIL}
    out = subprocess.run(["git", "commit-tree", tree], cwd=ROOT, input=MESSAGE, capture_output=True,
                         text=True, env={**os.environ, **who})
    if out.returncode != 0:
        raise SystemExit(f"git commit-tree failed: {out.stderr.strip()}")
    root = out.stdout.strip()
    git("update-ref", TEMP_REF, root)
    try:
        repo.mkdir(parents=True)
        git("init", "--quiet", "-b", "main", cwd=repo)
        git("fetch", "--quiet", "--no-tags", str(ROOT), TEMP_REF, cwd=repo)
        git("checkout", "--quiet", "-B", "main", "FETCH_HEAD", cwd=repo)
    finally:
        git("update-ref", "-d", TEMP_REF)
    git("config", "user.name", OWNER_NAME, cwd=repo)
    git("config", "user.email", OWNER_EMAIL, cwd=repo)
    # checks: one commit, the audited tree, the owner's identity, no co-author line, nothing else
    body = git("log", "-1", "--format=%B", cwd=repo)
    checks = {
        "one commit": git("rev-list", "--all", "--count", cwd=repo) == "1",
        "tree = the audited tree": git("rev-parse", "HEAD^{tree}", cwd=repo) == tree,
        "author and committer = owner, no-reply": git("log", "-1", "--format=%ae|%ce", cwd=repo)
        == f"{OWNER_EMAIL}|{OWNER_EMAIL}",
        "no co-author line": "co-authored-by" not in body.lower(),
        "no remote": git("remote", cwd=repo) == "",
        "clean checkout": git("status", "--porcelain", cwd=repo) == "",
        "objects intact": subprocess.run(["git", "fsck", "--no-progress"], cwd=repo,
                                         capture_output=True).returncode == 0,
    }
    return {"commit": git("rev-parse", "HEAD", cwd=repo), "tree": tree, "from_head": head,
            "files": len(git("ls-files", cwd=repo).splitlines()), "checks": checks}


def build_files(files: Path, model: Path, bloom: Path, repo: Path) -> dict:
    attach = files / "attach"
    attach.mkdir(parents=True)
    shutil.copyfile(model, attach / MODEL)
    shutil.copyfile(bloom, attach / FILTER)
    sums = {MODEL: _sha256(attach / MODEL), FILTER: _sha256(attach / FILTER)}
    checks = {f"{MODEL} is the published model": sums[MODEL] == PUBLISHED_WEIGHTS_SHA256,
              f"{FILTER} is the published filter": _content_sha256(attach / FILTER) == PUBLISHED_KNOWN_URLS_SHA256}
    (attach / "SHA256SUMS.txt").write_text("".join(f"{s}  {n}\n" for n, s in sums.items()),
                                          encoding="utf-8", newline="\n")
    size = {n: (attach / n).stat().st_size for n in sums}
    (files / "RELEASE_NOTES.md").write_text(f"""# Limnex UIE {TAG}

The first public release of the **Limnex URL Intelligence Engine**: a character-level transformer,
trained from scratch, that writes synthetic benign and phishing-style URLs for defensive research.

**Read the README's [Responsible Use]({PUBLIC_REPO}#responsible-use) and
[No Liability]({PUBLIC_REPO}#no-liability) sections before any use.**

| file | what it is | bytes | sha256 |
|---|---|---|---|
| `{MODEL}` | the model (engine `e16_b5`) | {size[MODEL]:,} | `{sums[MODEL]}` |
| `{FILTER}` | the known-URL filter; every generation requires it | {size[FILTER]:,} | `{sums[FILTER]}` |
| `SHA256SUMS.txt` | both fingerprints, for `sha256sum -c SHA256SUMS.txt` | | |

`uie.load` checks both files before use. Start with the README's *Quick start*; the model is
described in `MODEL_CARD.md`, and `REPRODUCE.md` shows how to check the published results.
Licensed under the Apache License 2.0.
""", encoding="utf-8", newline="\n")
    (files / "LAUNCH_STEPS.md").write_text(f"""# Launch day: about 10 minutes, only after the legal read

Until step 4 nothing is public, and everything can still be changed or deleted.

1. **Create the repository, private for now.** On github.com: **New repository**, name
   `limnex-uie`, **Private**, and leave README, .gitignore and license all unticked (it must start
   empty). **Create repository**.
2. **Push the release.** In a terminal:

       cd {repo}
       git remote add origin {PUBLIC_REPO}.git
       git push -u origin main

3. **Create the release.** On the repository page: **Releases**, **Draft a new release**, tag
   `{TAG}` (**Create new tag on publish**), title `Limnex UIE {TAG}`, paste the text of
   `RELEASE_NOTES.md`, drag in the three files of the `attach` folder, **Publish release**.
4. **Look it over, then go public.** Read the repository page and the release page as a
   stranger would. When you are happy: **Settings**, **Change visibility**, **Public**.
5. **Final check:** ask Claude to download both files from the public release page and check
   their fingerprints.
""", encoding="utf-8", newline="\n")
    return {"sha256": sums, "bytes": size, "checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser(description="D2 step 5: build the release candidate")
    ap.add_argument("--repo", required=True, help="the new folder for the public repository")
    ap.add_argument("--files", required=True, help="the new folder for the release files")
    ap.add_argument("--model", default=str(ROOT / "data/e16/train/e16_cap/final.pt"))
    ap.add_argument("--bloom", default=str(ROOT / "data/bundle2/known_urls.bloom.gz"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    repo, files = Path(args.repo).resolve(), Path(args.files).resolve()
    for p in (repo, files):
        if p.exists():
            raise SystemExit(f"{p} already exists: the release is built into new folders only")
    r = build_repo(repo)
    f = build_files(files, Path(args.model), Path(args.bloom), repo)
    print(f"repository {repo}: commit {r['commit'][:12]}, {r['files']} files, tree {r['tree'][:12]} "
          f"(= this repository's HEAD {r['from_head'][:12]})")
    print(f"release files {files}:")
    for n, s in f["sha256"].items():
        print(f"  {n}  {f['bytes'][n]:,} bytes  sha256 {s}")
    ok = True
    for k, v in {**r["checks"], **f["checks"]}.items():
        ok &= v
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("RELEASE CANDIDATE READY" if ok else "RELEASE CANDIDATE NOT READY")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
