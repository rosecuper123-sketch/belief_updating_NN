"""Write a machine-readable provenance record for the twelve main Stan fits.

Why this exists
---------------
Nothing under `metrics/` or `fits/` recorded which episode *draw* a fit came from
or which sampler settings produced it.  The mac2B group was re-drawn twice and
re-fit under a non-default `adapt_delta`, and that fact lived only in prose
(README 3.9/3.10) and in the external CmdStan runtime directory.  The audit
(CODE_AND_RESULTS_AUDIT.md, Major) asked for draw seed, source-trial hash, fit
hash and sampler settings to be written down; this script does exactly that.

It changes no fitting: it only reads existing artifacts and hashes them.

How each field is determined
----------------------------
* draw_seed  -- by matching this group's `episode_id` sequence in
                `metrics/m3_sampling_manifest.csv` against the candidate
                manifests (`metrics/_archived_20260930/` and
                `redraw/draw_<seed>/`).  A match is required; otherwise the
                group is reported as `unknown` rather than guessed.
* sampler    -- read from the CmdStan config JSON under
                `$M31_RUNTIME_ROOT/<model>_condA_<group>_<mode>/cmdstan_csv/`.
                If that directory is absent (it lives outside the repo), the
                values fall back to the documented defaults and the entry is
                flagged `assumed_default`.
* hashes     -- sha256 of data/*.rds, fits/.../fit.rds and the source
                metrics/..._m3_subjects.csv.

Outputs (overwritten on each run)
    metrics/fit_provenance.json
    metrics/fit_provenance.csv

Usage:
    python scripts/write_fit_provenance.py
    python scripts/write_fit_provenance.py --check     # verify, write nothing
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

FITTING = Path(__file__).resolve().parent.parent
GROUPS = ("mac1A", "mac1B", "mac2A", "mac2B")
MODELS = ("M1", "M2", "M3")
MODE = "final"
ARM = "h16_b200"

DEFAULT_SAMPLER = {"adapt_delta": 0.95, "max_treedepth": 14,
                   "chains": 4, "warmup": 2000, "samples": 2000, "seed": 13283435}
RUNTIME_ROOT = Path(os.environ.get("M31_RUNTIME_ROOT", "D:/m31_runtime_h16"))


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: Path) -> dict[str, tuple[int, ...]]:
    """group -> episode_id sequence, in subject_index order."""
    if not path.exists():
        return {}
    out: dict[str, list[tuple[int, int]]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["group"], []).append(
                (int(row["subject_index"]), int(row["episode_id"]))
            )
    return {g: tuple(e for _, e in sorted(v)) for g, v in out.items()}


def candidate_draws() -> dict[int, dict[str, tuple[int, ...]]]:
    """draw seed -> {group: episode_id sequence}."""
    cands: dict[int, dict[str, tuple[int, ...]]] = {}
    archived = FITTING / "metrics" / "_archived_20260930" / "m3_sampling_manifest.csv"
    if archived.exists():
        cands[20260930] = read_manifest(archived)
    for d in sorted((FITTING / "redraw").glob("draw_*")):
        try:
            seed = int(d.name.split("_")[-1])
        except ValueError:
            continue
        m = read_manifest(d / "metrics" / "m3_sampling_manifest.csv")
        if m:
            cands[seed] = m
    return cands


def sampler_for(model: str, group: str) -> dict:
    tag = f"{model}_condA_{group}_{MODE}"
    csv_dir = RUNTIME_ROOT / tag / "cmdstan_csv"
    if not csv_dir.is_dir():
        return {**DEFAULT_SAMPLER, "source": "assumed_default"}
    configs = sorted(csv_dir.glob("*-*_config.json"))
    if not configs:
        return {**DEFAULT_SAMPLER, "source": "assumed_default"}
    cfg = json.loads(configs[0].read_text(encoding="utf-8"))
    sample = cfg.get("method", {}).get("sample", {})
    nuts = sample.get("algorithm", {}).get("hmc", {}).get("engine", {}).get("nuts", {})
    out = {
        "adapt_delta": sample.get("adapt", {}).get("delta"),
        "max_treedepth": nuts.get("max_depth"),
        "chains": len(configs),
        "warmup": sample.get("num_warmup"),
        "samples": sample.get("num_samples"),
        "seed": cfg.get("random", {}).get("seed"),
        "start_datetime": cfg.get("start_datetime"),
        "source": "cmdstan_config",
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify only, write nothing")
    args = parser.parse_args()

    active = read_manifest(FITTING / "metrics" / "m3_sampling_manifest.csv")
    if not active:
        raise SystemExit(f"cannot read {FITTING / 'metrics' / 'm3_sampling_manifest.csv'}")
    cands = candidate_draws()
    if not cands:
        raise SystemExit("no candidate draw manifests found (metrics/_archived_*, redraw/draw_*)")

    entries: list[dict] = []
    for group in GROUPS:
        seq = active.get(group)
        if seq is None:
            raise SystemExit(f"group {group} missing from the active manifest")
        matches = [seed for seed, m in cands.items() if m.get(group) == seq]
        draw_seed = matches[0] if len(matches) == 1 else None
        if draw_seed is None:
            print(f"WARNING: {group} episode sequence matched {len(matches)} candidates "
                  f"({matches}); reporting draw_seed=unknown", file=sys.stderr)

        for model in MODELS:
            s = sampler_for(model, group)
            entries.append({
                "model": model,
                "group": group,
                "mode": MODE,
                "arm": ARM,
                "draw_seed": draw_seed,
                "sampler": s,
                "hashes": {
                    "data_rds": sha256(FITTING / "data" / f"formal_osla__{ARM}__{group}.rds"),
                    "fit_rds": sha256(FITTING / "fits" / f"{model}_condA_{group}_{MODE}" / "fit.rds"),
                    "trials_csv": sha256(FITTING / "metrics" / f"formal_osla__{ARM}__{group}_m3_subjects.csv"),
                },
            })

    payload = {
        "fitting_dir": str(FITTING),
        "runtime_root": str(RUNTIME_ROOT),
        "runtime_root_exists": RUNTIME_ROOT.is_dir(),
        "draw_seed_method": "exact match of the group's episode_id sequence against candidate manifests",
        "candidate_manifests": {str(k): v for k, v in sorted(cands.items())},
        "note": ("mac2B uses a selected draw (chosen because it converged) and a non-default "
                 "adapt_delta; it is therefore not sampler-comparable to mac1A/mac1B/mac2A."),
        "entries": entries,
    }

    print(f"{'model':<4} {'group':<6} {'draw':<9} {'delta':<6} {'depth':<6} {'chains':<7} source")
    for e in entries:
        s = e["sampler"]
        print(f"{e['model']:<4} {e['group']:<6} {str(e['draw_seed']):<9} "
              f"{str(s.get('adapt_delta')):<6} {str(s.get('max_treedepth')):<6} "
              f"{str(s.get('chains')):<7} {s.get('source')}")

    if args.check:
        print("\n--check: nothing written")
        return

    out_json = FITTING / "metrics" / "fit_provenance.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    cols = ["model", "group", "mode", "arm", "draw_seed", "adapt_delta", "max_treedepth",
            "chains", "warmup", "samples", "seed", "sampler_source",
            "data_rds_sha256", "fit_rds_sha256", "trials_csv_sha256"]
    out_csv = FITTING / "metrics" / "fit_provenance.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for e in entries:
            s, h = e["sampler"], e["hashes"]
            w.writerow([e["model"], e["group"], e["mode"], e["arm"], e["draw_seed"],
                        s.get("adapt_delta"), s.get("max_treedepth"), s.get("chains"),
                        s.get("warmup"), s.get("samples"), s.get("seed"), s.get("source"),
                        h["data_rds"], h["fit_rds"], h["trials_csv"]])

    print(f"\nwrote {out_json}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    sys.exit(main())
