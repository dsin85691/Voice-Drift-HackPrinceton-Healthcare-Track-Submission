"""Data preparation (spec 3.1). CPU runtime. Interviewee turns only.

Androids layout (verified 2026-11-14 on the real download):
  - Interview-Task/audio_clip/<speaker>/<speaker>_<k>.wav: one clip per participant turn; clip durations match
    the start/end pairs in interview_timedata.csv, so the clips are the participant's speech (spec: interviewee only).
  - fold-lists.csv (plural): two header rows, Read folds in cols 0-4, Interview folds in cols 7-11.
  - Label comes from the speaker id nn_XGmm_t (X = P/C). The zip also contains __MACOSX junk, which is skipped.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

WIN_S, MIN_TAIL_S, MAX_PER_SPK = 8.0, 4.0, 40


def is_junk(p: Path) -> bool:
    """macOS zip artefacts: __MACOSX/ folders and ._* AppleDouble stubs are not audio."""
    return "__MACOSX" in p.parts or p.name.startswith("._")


def speaker_from_path(p: Path) -> str:
    """Androids layout: Interview-Task/audio_clip/<nn_XGmm_t>/<nn_XGmm_t>_<k>.wav.
    The full folder name is the speaker id: `nn` alone repeats across the C (control) and P (patient) groups."""
    if p.stem.startswith(p.parent.name + "_"):
        return p.parent.name
    return p.stem.rsplit("_", 1)[0] if p.stem.count("_") >= 3 else p.stem


def label_from_speaker(spk: str) -> int | None:
    """nn_XGmm_t: X = P (patient, 1) or C (control, 0). README: X is missing for 2 participants -> None."""
    part = spk.split("_")[1] if "_" in spk else ""
    return {"P": 1, "C": 0}.get(part[:1]) if len(part) == 4 else None


def window_spans(duration_s: float, win_s: float = WIN_S, min_tail_s: float = MIN_TAIL_S) -> list[tuple[float, float]]:
    """Non-overlapping windows inside one clip; tails shorter than 4 s are dropped."""
    out, t = [], 0.0
    while t < duration_s - 1e-9:
        e = min(t + win_s, duration_s)
        if e - t >= win_s - 1e-9 or e - t >= min_tail_s:
            out.append((round(t, 3), round(e, 3)))
        t += win_s
    return out


def cap_windows(rows: list[dict], cap: int = MAX_PER_SPK, seed: int = 0) -> list[dict]:
    """Seeded cap per speaker; deterministic for reruns."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["speaker_id"], []).append(r)
    kept = []
    for spk in sorted(by):
        rs = by[spk]
        if len(rs) > cap:
            rs = random.Random(f"{seed}|{spk}").sample(rs, cap)
        kept += sorted(rs, key=lambda r: (r["src_file"], r["t0"]))
    return kept


def read_androids_folds(path: str, task: str = "Interview") -> dict[str, int]:
    """fold-lists.csv layout (verified on the real download):
        row 0: 'Read' over cols 0-4, 'Interview' over cols 7-11
        row 1: fold1..fold5 under each task
        rows 2+: quoted speaker ids, one column per fold (columns have different lengths).
    Returns {speaker_id: fold 1..5} for `task`. Raises if a speaker appears in two folds."""
    import csv

    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    start = next(i for i, c in enumerate(rows[0]) if c.strip().lower() == task.lower())
    cols = {}
    for j in range(start, len(rows[1])):
        name = rows[1][j].strip().lower()
        if not name.startswith("fold"):
            if cols:
                break
            continue
        cols[j] = int(name[4:])
    out: dict[str, int] = {}
    for r in rows[2:]:
        for j, fold in cols.items():
            spk = r[j].strip().strip("'\"") if j < len(r) else ""
            if not spk:
                continue
            if spk in out and out[spk] != fold:
                raise ValueError(f"{spk} listed in folds {out[spk]} and {fold}")
            out[spk] = fold
    return out


def build_manifest(clips_glob: str, fold_csv: str, out: str, seed: int = 0, task: str = "Interview"):
    """One row per 8 s window of each participant clip. src_file is ABSOLUTE so the manifest works from any cwd."""
    import glob as _glob

    import pandas as pd
    import soundfile as sf

    folds = read_androids_folds(fold_csv, task)
    rows, no_fold, no_label = [], set(), set()
    for f in sorted(_glob.glob(clips_glob, recursive=True)):   # glob.glob accepts absolute patterns
        p = Path(f).resolve()
        if is_junk(p):
            continue
        spk = speaker_from_path(p)
        lab = label_from_speaker(spk)
        if spk not in folds:
            no_fold.add(spk); continue
        if lab is None:
            no_label.add(spk); continue
        dur = sf.info(str(p)).duration
        for t0, t1 in window_spans(dur):
            rows.append({"window_id": f"{spk}:{p.stem}:{t0:.2f}", "speaker_id": spk, "label": lab,
                         "fold": folds[spk], "src_file": str(p), "t0": t0, "t1": t1})
    if no_fold or no_label:
        print(f"WARNING skipped speakers - not in {task} folds: {sorted(no_fold)}; no C/P label: {sorted(no_label)}")
    df = pd.DataFrame(cap_windows(rows, seed=seed))
    assert df.groupby("speaker_id").fold.nunique().max() == 1, "a speaker spans folds"
    df.to_parquet(out)
    print(f"{len(df)} windows | {df.speaker_id.nunique()} speakers | "
          f"per label {df.groupby('label').speaker_id.nunique().to_dict()} | per fold {df.groupby('fold').speaker_id.nunique().to_dict()}")
    return df


def split_noise(file_lists: dict[str, list[str]], seed: int = 0, test_frac: float = 0.2,
                caps: dict[str, int] | None = None) -> dict:
    """Split noise by FILE (never by window) into train / test pools.
    file_lists: {"noise": [...], "music": [...], "speech": [...], "rir_real": [...], "rir_point_noise": [...]}.
    The RIR database's real RIRs and point-source noises are test-only."""
    out: dict[str, list[str]] = {}
    for k in ("noise", "music", "speech"):
        fs = sorted(file_lists.get(k, []))
        random.Random(f"{seed}|{k}").shuffle(fs)
        if caps and caps.get(k):
            fs = fs[: caps[k]]          # seeded cap BEFORE the by-file split (a few hundred files is plenty)
        n_test = max(1, int(len(fs) * test_frac)) if fs else 0
        out[f"{k}_test"], out[f"{k}_train"] = fs[:n_test], fs[n_test:]
    out["rir_test"] = sorted(file_lists.get("rir_real", []))
    out["noise_test"] += sorted(file_lists.get("rir_point_noise", []))
    assert not set(out["noise_train"]) & set(out["noise_test"])
    return out


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-glob", required=True)
    ap.add_argument("--fold-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise-root", help="dir with musan/ and RIRS_NOISES/")
    ap.add_argument("--noise-out")
    ap.add_argument("--cap-noise", type=int, default=0)
    ap.add_argument("--cap-music", type=int, default=200)
    ap.add_argument("--cap-speech", type=int, default=300)
    a = ap.parse_args()
    df = build_manifest(a.clips_glob, a.fold_csv, a.out)
    print(len(df), "windows,", df.speaker_id.nunique(), "speakers")
    if a.noise_root:
        r = Path(a.noise_root)
        g = lambda pat: [str(p) for p in r.glob(pat)]
        idx = split_noise({"noise": g("musan/noise/**/*.wav"), "music": g("musan/music/**/*.wav"),
                           "speech": g("musan/speech/**/*.wav"), "rir_real": g("RIRS_NOISES/real_rirs_isotropic_noises/*rir*.wav"),
                           "rir_point_noise": g("RIRS_NOISES/pointsource_noises/*.wav")},
                          caps={"noise": a.cap_noise, "music": a.cap_music, "speech": a.cap_speech})
        Path(a.noise_out).write_text(json.dumps(idx))
        used = sorted({f for v in idx.values() for f in v})      # pack ONLY these into the Drive tar
        Path(a.noise_out).with_suffix(".files.txt").write_text("\n".join(used))
        print({k: len(v) for k, v in idx.items()}, "-> list of", len(used), "files")


if __name__ == "__main__":
    main()
