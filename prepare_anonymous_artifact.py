"""Check local evidence and create a curated anonymous review archive.

Run from any directory. Does not rerun allocators or change working sources.
The archive deliberately excludes Git history, old archives, logs, caches,
the experimental LNCS draft, and unreferenced manuscript assets.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "output/submission"


def main():
    checks = {}
    for manifest in sorted((ROOT / "data").glob("*.manifest.json")):
        info = json.loads(manifest.read_text())
        data = Path(str(manifest).removesuffix(".manifest.json"))
        assert hashlib.sha256(data.read_bytes()).hexdigest() == info["csv_sha256"], data.name
        with data.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == info["rows"], data.name
        checks[data.name] = {"sha256_matches": True, "rows": len(rows),
                             "represented_blocks": len({r["block"] for r in rows})}

    frontier = pd.read_csv(ROOT / "results/retrieval_frontier.csv")
    benchmark = pd.read_csv(ROOT / "results/retrieval_benchmark.csv")
    assert set(frontier.seed) == {0, 1, 2}
    assert set(frontier.method) == {"ALPS", "TXALLO", "METIS-SENDER", "SENDER-KMEANS-BAL"}
    matched = frontier[frontier.target_coverage.eq(.9)].merge(
        benchmark[benchmark.query_mode.eq("history90")],
        on=["trace", "method", "seed", "category"],
        suffixes=("_sweep", "_benchmark"), validate="one_to_one")
    assert len(matched) == 372
    for key in ("contacted_regions", "rows_scanned", "result_recall"):
        assert np.allclose(matched[key + "_sweep"], matched[key + "_benchmark"],
                           rtol=0, atol=1e-12), key
    complete = frontier[frontier.target_coverage.eq(1)]
    assert complete.result_recall.eq(1).all()
    assert complete.contacted_regions.eq(6).all()
    assert complete.rows_scanned.eq(complete.suffix_rows).all()
    assert frontier.groupby(["trace", "method", "seed", "target_coverage", "category"]).size().eq(1).all()
    checks["retrieval"] = {"sweep_rows": len(frontier),
                           "history90_rows_reproduced": len(matched),
                           "seeds": [0, 1, 2], "complete_controls_pass": True}

    texpath = ROOT / "ALPS_conference_submission/main.tex"
    tex = texpath.read_text(encoding="utf-8")
    assert "\\anonymoustrue" in tex
    identities = re.findall(r"pdfauthor=\{(?!Anonymous\})([^}]+)\}", tex)
    # Remove complete inactive branches, rather than hiding identity behind a flag.
    tex = re.sub(r"\\ifanonymous\s*(.*?)\\else\s*.*?\\fi",
                 lambda m: m.group(1), tex, flags=re.S)
    assert not any(name in tex for name in identities)
    entries = {}
    for p in sorted(ROOT.glob("*.py")):
        entries[p.relative_to(ROOT).as_posix()] = p.read_bytes()
    for folder, suffixes in (("sim", {".py"}), ("tests", {".py"}),
                             ("data", {".csv", ".json"}),
                             ("results", {".csv", ".png", ".svg", ".json", ".jsonl"})):
        for p in sorted((ROOT / folder).rglob("*")):
            if p.is_file() and p.suffix in suffixes:
                entries[p.relative_to(ROOT).as_posix()] = p.read_bytes()
    for name in ("README.md", "ARTIFACT_VERIFICATION.md", "NEW_VALIDATION_RESULTS.md",
                 "REPRODUCE_CLEAN.md", "CLEAN_REPRODUCTION.json",
                 "SEMANTIC_SCALE_VERIFICATION.json",
                 "requirements.txt", "requirements-metis.txt",
                 "requirements-core-lock.txt", "environment-metis-lock.yml"):
        entries[name] = (ROOT / name).read_bytes()
    entries["ALPS_conference_submission/main.tex"] = tex.encode("utf-8")
    entries["ALPS_conference_submission/references.bib"] = texpath.with_name("references.bib").read_bytes()
    for asset in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        p = texpath.parent / asset
        entries[p.relative_to(ROOT).as_posix()] = p.read_bytes()
    paper_pdf = texpath.with_name("main.pdf")
    pdf = PdfReader(paper_pdf)
    assert len(pdf.pages) >= 1
    assert pdf.metadata.get("/Author") == "Anonymous"
    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    assert "Anonymous Author(s)" in text and "??" not in text
    entries["ALPS_conference_submission/main.pdf"] = paper_pdf.read_bytes()
    # Scan curated text and embedded PDF metadata for known local identifiers.
    forbidden = tuple(name.lower() for name in identities) + ("c:\\users\\", "c:/users/")
    for name, data in entries.items():
        if Path(name).suffix in {".tex", ".md", ".py", ".bib", ".json", ".csv", ".txt"}:
            content = data.decode("utf-8-sig").lower()
            # This checking utility legitimately contains the scanner's patterns.
            if name != Path(__file__).name:
                assert not any(x in content for x in forbidden), name
        if name.endswith(".pdf"):
            import io
            meta = str(PdfReader(io.BytesIO(data)).metadata).lower()
            assert not any(x in meta for x in forbidden), name
    checks["pdf"] = {"pages": len(pdf.pages), "author": "Anonymous", "unresolved_markers": False}
    entries["LOCAL_CHECKS.json"] = (json.dumps(checks, indent=2) + "\n").encode()
    inventory = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(entries.items())}
    entries["SHA256SUMS.json"] = (json.dumps(inventory, indent=2) + "\n").encode()
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "ALPS_anonymous_artifact.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(entries.items()):
            # Neutral timestamps also avoid disclosing local file creation times.
            item = zipfile.ZipInfo("ALPS/" + name, (2026, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(item, data)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, digest in inventory.items():
            assert hashlib.sha256(z.read("ALPS/" + name)).hexdigest() == digest
    print(json.dumps(checks, indent=2))
    print(f"Verified {len(entries)} archive entries: {archive}")


if __name__ == "__main__":
    main()
