# %% [markdown]
# # Voice Drift Monitor: Colab run
# Open this file in VS Code and pick a **Colab** kernel (Colab extension). Run cells top to bottom.
# CPU runtime for sections 0, 1 and 3; **L4 GPU runtime only for section 2** (G0 -> G1 -> G2).
# After each GPU stage the budget wrapper unassigns the runtime; reconnect to a CPU runtime for section 3.

# %% 0. Setup (CPU runtime)
import os, subprocess, sys
from google.colab import drive
drive.mount("/content/drive")
VD = "/content/drive/MyDrive/vd"                  # repo + data + feats + budget/ + artifacts/ live here
REPO = f"{VD}/repo"                               # git clone / copy of this project
os.chdir(REPO); sys.path.insert(0, REPO)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas", "pyarrow", "soundfile", "transformers"], check=True)
subprocess.run([sys.executable, "tools/budget.py", "--drive", VD], check=True)   # prints hours used / remaining

# %% 1. Data prep (CPU). EDIT the globs/columns after inspecting the real Androids layout (spec section 13).
ANDROIDS = f"{VD}/data/androids"
# !ls {ANDROIDS}     # look first: clip folders, turn-timing file, fold-list.csv columns
CLIPS_GLOB = f"{ANDROIDS}/**/*.wav"
FOLD_CSV = f"{ANDROIDS}/fold-list.csv"
from research import data_prep
df = data_prep.build_manifest(CLIPS_GLOB, FOLD_CSV, f"{VD}/manifest.parquet")   # cols=(speaker, label, fold) if names differ
print(len(df), "windows;", df.speaker_id.nunique(), "speakers; expect ~4,600 windows")

# %% 1b. Noise split by file (MUSAN + OpenSLR-28 must already be extracted under NOISE_ROOT)
NOISE_ROOT = f"{VD}/data/noise"
# !python -m research.data_prep --clips-glob "{CLIPS_GLOB}" --fold-csv {FOLD_CSV} --out {VD}/manifest.parquet --noise-root {NOISE_ROOT} --noise-out {VD}/noise_index.json

# %% 2. GPU stages. Switch runtime to L4 FIRST. Run ONE stage per runtime session.
STAGE = "G0"        # G0 (smoke) -> G1 -> G2. Check G0 throughput >= 20 clips/s before G1.
LIMIT = "200" if STAGE == "G0" else "0"
subprocess.run([sys.executable, "-m", "research.extract", "--stage", STAGE, "--limit", LIMIT,
                "--manifest", f"{VD}/manifest.parquet", "--noise-index", f"{VD}/noise_index.json",
                "--out", f"{VD}/feats"], check=True)

# %% L. LIGHTWEIGHT PROJECTION RUN (projection-head spec v0.1) - the fast path, replaces G1/G2/T1 for the demo
# L-a (CPU runtime, once): pack audio into two files on Drive
#   !tar -cf data/androids_clips.tar -C data/androids/Androids-Corpus Interview-Task/audio_clip
#   !tar -cf data/noise_subset.tar -C data noise_subset
# L-b (L4 runtime): unpack to local disk, then extract 480 windows x 5 views (~2,400 clips) in the background
#   !mkdir -p /content/data/androids_clips && tar -xf data/androids_clips.tar -C /content/data/androids_clips && tar -xf data/noise_subset.tar -C /content/data
#   PM = ("--path-map /content/drive/MyDrive/vd/repo/data/androids/Androids-Corpus=/content/data/androids_clips "
#         "--path-map /content/drive/MyDrive/vd/repo/data/noise_subset=/content/data/noise_subset")
#   !nohup python -m research.lite.extract --manifest manifest.parquet --noise-index noise_index.json --out lite_feats {PM} > lite_feats.log 2>&1 &
#   !tail -3 lite_feats.log          # after ~1 min: clips/s; stop with  !pkill -f research.lite.extract
# L-c (any runtime, minutes): train B1 + P1, evaluate B0/B1/P1, report, plots, checkpoints, dashboard export
#   from research.lite import train as LT
#   res, runs, adir = LT.run_all("lite_feats", out_dir="artifacts", version=1, layer=6)
#   -> reports/lite_report.md, reports/lite_plots.png, reports/checkpoints/{B1,P1}.pt, artifacts/v1/

# %% 3. Train + evaluate + export, ON GPU (T1). Switch to an L4/T4 runtime first.
# This is OUTSIDE the spec's GPU plan (G0-G3): it spends reserve hours, so get the team lead's OK.
# The budget wrapper meters it as stage "T1" (planned 0.5 h, kill 0.75 h) and UNASSIGNS the runtime on exit,
# so everything must be written to Drive inside the block. Variables are lost afterwards.
import json
from research import budget, evaluate, export, load_cache, train_heads
with budget.stage("T1", VD):
    d = load_cache.load_data(f"{VD}/feats", f"{VD}/manifest.parquet")
    print(d.Zc.shape, "windows;", len(set(d.spk)), "speakers")
    res, chosen, oof = train_heads.run_cv(d, models=("L0", "M1", "M3"), device="cuda", return_oof=True)
    print("layers picked per fold:", chosen)
    e1 = evaluate.e1_table(res)             # L0 vs M1 vs M3 at clean/20/10/5 dB, paired CI (M3-L0)
    # GATE 1: if every L0 AUC CI includes 0.5, apply the section 11 fallbacks before shipping.
    for r in e1: print(r)
    json.dump(e1, open(f"{VD}/e1_table.json", "w"), indent=1, default=float)       # persist before unassign
    # Platt calibration of the deployed head (M3) on OUT-OF-FOLD 3-window session logits
    from research import calibration
    S, Y, F, Z = calibration.session_logits(oof["M3"], d.spk, d.y, d.fold)
    cal = calibration.fit_calibration(S, Y, F, Z, model="M3")
    print("calibration:", {k: cal["metrics"][k] for k in ("auc", "auc_ci", "ece", "ece_uncalibrated", "brier")},
          "gate_passed:", cal["gate_passed"])
    json.dump(cal, open(f"{VD}/calibration.json", "w"), indent=1, default=float)
    out_dir, cfg = export.export(d, chosen, f"{VD}/artifacts", version=1,
                                 metrics={"E1": e1, "layers": chosen, "calibration": cal["metrics"]},
                                 wavlm_revision="PIN_ME", device="cuda", calibration=cal)
    print("exported to", out_dir, "tau =", cfg["tau"])
# (CPU alternative: drop the budget wrapper and device="cuda"; same code.)
# TODO E2/E3: need the trained projector + per-control-speaker features; not wired up yet.
# TODO E3: build per-control-speaker window features in raw-z and P(z) space and call evaluate.e3_false_alerts(...)
