"""Run the BatteryLife replication experiments as a resumable queue.

Every run is built from the authors' own script in upstream/train_eval_scripts:
the script is copied, a few variables are overridden (dataset, seed, selected
hyperparameters, paths, GPUs) and the copy is executed from upstream/.

Each run writes to results/<data>/<table>/<model>/<group>/<seed>/:
    run.sh        the exact script that was executed
    run.log       full console output
    run.json      settings, commits, data version, GPU info, timing, exit code
    metrics.json  parsed final metrics (written only when the run succeeds)

A run whose metrics.json exists is skipped, so the queue can be stopped and
restarted at any time.

Examples (from the repo root):
    python scripts/run_queue.py --list
    python scripts/run_queue.py --only "main/CPMLP/" --gpus 0,1
    python scripts/run_queue.py --gpus 0,1 --max-hours 10
    python scripts/run_queue.py --only "main/CPMLP/CALB/2021" --dry-run
"""
import argparse
import datetime as dt
import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "upstream"
SCRIPTS = UPSTREAM / "train_eval_scripts"
UPSTREAM_COMMIT = "3b17bb5"  # Ruifeng-Tan/BatteryLife commit copied into upstream/

SEEDS = [42, 2021, 2024]
GROUPS = ["Li-ion", "Zn-ion", "Na-ion", "CALB"]
# Dataset name per group and seed (assets/Model_training.md).
DATASET = {
    "Li-ion": {42: "MIX_large", 2021: "MIX_large", 2024: "MIX_large"},
    "Zn-ion": {42: "ZN-coin42", 2021: "ZN-coin", 2024: "ZN-coin2024"},
    "Na-ion": {42: "NAion42", 2021: "NAion", 2024: "NAion2024"},
    "CALB": {42: "CALB42", 2021: "CALB", 2024: "CALB2024"},
}
DUMMY_NAME = {"Li-ion": "MIX_large", "Zn-ion": "ZN-coin", "Na-ion": "NAion", "CALB": "CALB"}

SELECTED_MODELS = ["CPMLP", "CPTransformer", "CPGRU", "CPLSTM"]
SCRIPT_ONLY_CP = ["CPBiGRU", "CPBiLSTM"]
BASELINES = ["DLinear", "MLP", "PatchTST", "Autoformer", "iTransformer", "CNN", "MICN"]


def selected_hyperparameters():
    """Parse the table in upstream/assets/Selected_hyperparameters.md."""
    table = {}
    text = (UPSTREAM / "assets" / "Selected_hyperparameters.md").read_text()
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 10 or cells[0] in ("model", "") or set(cells[0]) <= set("-:"):
            continue
        model, group, seed, bs, dm, dff, el, dl, drop, lr = cells
        table[(model, group, int(seed))] = {
            "batch_size": bs, "d_model": dm, "d_ff": dff, "e_layers": el,
            "d_layers": dl, "dropout": drop, "learning_rate": lr,
        }
    return table


def build_runs():
    """All runs in execution order. Each run is a dict."""
    hp = selected_hyperparameters()
    runs = []

    def add(table, model, group, seed, kind, script, overrides=None, needs=None):
        runs.append({"id": f"{table}/{model}/{group}/{seed}", "table": table, "model": model,
                     "group": group, "seed": seed, "kind": kind, "script": script,
                     "overrides": overrides or {}, "needs": needs})

    def selected(model, group, seed):
        o = dict(hp[(model, group, seed)])
        if model in ("CPGRU", "CPLSTM"):
            # These models read their recurrent depth from lstm_layers.
            o["lstm_layers"] = o["d_layers"]
        return o

    # A. Dummy
    for group in GROUPS:
        for seed in SEEDS:
            add("main", "Dummy", group, seed, "dummy", None)
    # B. CP models: CPMLP and CPTransformer first, then the other CP models
    for model in SELECTED_MODELS + SCRIPT_ONLY_CP:
        for group in GROUPS:
            for seed in SEEDS:
                o = selected(model, group, seed) if model in SELECTED_MODELS else {}
                add("main", model, group, seed, "train", f"{model}.sh", o)
    # C. Baselines with the values in their scripts
    for model in BASELINES:
        for group in GROUPS:
            for seed in SEEDS:
                add("main", model, group, seed, "train", f"{model}.sh")
    # D. Ablation: drop the intra-cycle (e_layers) or inter-cycle (d_layers) encoder
    for model in ("CPMLP", "CPTransformer"):
        for variant, key in (("wo-intra", "e_layers"), ("wo-inter", "d_layers")):
            for group in GROUPS:
                for seed in SEEDS:
                    o = selected(model, group, seed)
                    o[key] = "0"
                    add("ablation", f"{model}-{variant}", group, seed, "train", f"{model}.sh", o)
    # F. Transfer from the Li-ion CPMLP of the same seed
    for group in ("Zn-ion", "Na-ion", "CALB"):
        for seed in SEEDS:
            src = f"main/CPMLP/Li-ion/{seed}"
            add("transfer", "Frozen", group, seed, "eval", "evaluate.sh",
                {"eval_cycle_min": "1", "eval_cycle_max": "100", "model": "CPMLP"}, needs=src)
            add("transfer", "FT", group, seed, "finetune", "finetune_script.sh",
                {"batch_size": "4"}, needs=src)
            add("transfer", "DA", group, seed, "da", "domain_adaptation_script.sh",
                {"dataset": "MIX_large"})
    return runs


def set_var(text, key, value):
    """Set a shell variable everywhere it is assigned, or define it at the top."""
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.M)
    line = f"{key}={value}"
    if pattern.search(text):
        return pattern.sub(lambda _: line, text)
    return line + "\n" + text


def checkpoint_dir(args, run_id):
    return args.checkpoints / args.data / run_id


def find_trained_model(args, run_id):
    hits = sorted(checkpoint_dir(args, run_id).glob("*/args.json"))
    return hits[-1].parent if hits else None


def make_script(args, run, port):
    if run["kind"] == "dummy":
        name = DUMMY_NAME[run["group"]]
        return f"python models/Dummy.py {name} {run['seed']}\n"

    text = (SCRIPTS / run["script"]).read_text()
    target = DATASET[run["group"]][run["seed"]]
    o = dict(run["overrides"])
    o["num_process"] = str(len(args.gpu_list))
    o["master_port"] = str(port)
    o["checkpoints"] = str(checkpoint_dir(args, run["id"]))
    o["root_path"] = f"../dataset/{args.data}"

    if run["kind"] == "train":
        o["dataset"] = target
        o["seed"] = str(run["seed"])
    elif run["kind"] in ("finetune", "eval"):
        src = find_trained_model(args, run["needs"])
        if src is None:
            raise RuntimeError(f"needs a finished {run['needs']} checkpoint")
        o["args_path"] = f"{src}/"
        o["finetune_dataset" if run["kind"] == "finetune" else "eval_dataset"] = target
    elif run["kind"] == "da":
        o["target_dataset"] = target
        o["seed"] = str(run["seed"])

    for key, value in o.items():
        if key in ("checkpoints", "root_path") and run["kind"] == "eval":
            continue  # evaluate.sh takes paths from the trained model's args.json
        text = set_var(text, key, value)

    gpus = ",".join(args.gpu_list)
    text = re.sub(r"CUDA_VISIBLE_DEVICES=[0-9,]+", f"CUDA_VISIBLE_DEVICES={gpus}", text)
    if len(args.gpu_list) == 1:
        text = text.replace("--multi_gpu", "")
    return text


METRIC_PATTERNS = {
    "test_MAPE": r"Test MAPE: ([-\d.eE+]+)",
    "test_acc15": r"Test 15%-accuracy: ([-\d.eE+]+)",
    "test_acc10": r"Test 10%-accuracy: ([-\d.eE+]+)",
    "val_MAPE": r"Val MAPE: ([-\d.eE+]+)",
    "val_acc15": r"Val 15%-accuracy: ([-\d.eE+]+)",
    "test_seen_MAPE": r"Test Seen MAPE: ([-\d.eE+]+)",
    "test_unseen_MAPE": r"Test Unseen MAPE: ([-\d.eE+]+)",
    "test_seen_acc15": r"Test Seen 15%-accuracy: ([-\d.eE+]+)",
    "test_unseen_acc15": r"Test Unseen 15%-accuracy: ([-\d.eE+]+)",
}


def parse_metrics(kind, log):
    """Return metrics with accuracies as fractions (the code prints percentages)."""
    m = {}
    if kind == "dummy":
        for key, pat in (("test_MAPE", r"Test MAPE: ([-\d.eE+]+)"),
                         ("val_MAPE", r"Vali MAPE: ([-\d.eE+]+)"),
                         ("test_acc15", r"\| 0\.15_acc: ([-\d.eE+]+)")):
            hits = re.findall(pat, log)
            if hits:
                m[key] = float(hits[-1])
    elif kind == "eval":
        hits = re.findall(r"Eval cycle: 1-100 \| MAPE: ([-\d.eE+]+) \| [\d.]+-accuracy: ([-\d.eE+]+)%", log)
        if hits:
            m["test_MAPE"], m["test_acc15"] = float(hits[-1][0]), float(hits[-1][1])
    else:
        best = "\n".join(l for l in log.splitlines() if "Best model performance" in l)
        for key, pat in METRIC_PATTERNS.items():
            hits = re.findall(pat, best)
            if hits:
                m[key] = float(hits[-1])
    for key in list(m):
        if "acc" in key and m[key] >= 0:
            m[key] /= 100.0
    return m if "test_MAPE" in m else None


def gpu_info(gpu_list):
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = [dict(zip(("index", "name", "memory_total", "memory_used", "driver"),
                     [c.strip() for c in l.split(",")])) for l in out.strip().splitlines()]
    return [r for r in rows if r["index"] in gpu_list]


def free_mb(gpu_list):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    free = dict(l.split(", ") for l in out.strip().splitlines())
    return {g: int(free.get(g, 0)) for g in gpu_list}


def repo_commit():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="v4", choices=["v4", "v12"])
    p.add_argument("--gpus", default="0,1", help="GPU indices to use, e.g. 0,1")
    p.add_argument("--only", default="", help="regex on run ids, e.g. 'main/CPMLP/'")
    p.add_argument("--list", action="store_true", help="list runs and their status, then exit")
    p.add_argument("--dry-run", action="store_true", help="write run.sh files but do not run them")
    p.add_argument("--max-hours", type=float, default=None, help="do not start new runs after this many hours")
    p.add_argument("--min-free-mb", type=int, default=8000, help="stop if a GPU has less free memory")
    p.add_argument("--port", type=int, default=29500, help="base port for accelerate")
    p.add_argument("--results", type=Path, default=REPO / "results")
    p.add_argument("--checkpoints", type=Path, default=REPO / "checkpoints")
    args = p.parse_args()
    args.gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]

    runs = [r for r in build_runs() if re.search(args.only, r["id"])]
    out_root = args.results / args.data

    if args.list:
        done = 0
        for r in runs:
            ok = (out_root / r["id"] / "metrics.json").exists()
            done += ok
            print(f"{'done' if ok else '    '}  {r['id']}")
        print(f"{done}/{len(runs)} done")
        return

    start = time.time()
    for i, r in enumerate(runs):
        out = out_root / r["id"]
        if (out / "metrics.json").exists():
            continue
        if args.max_hours is not None and time.time() - start > args.max_hours * 3600:
            print(f"Stopping: --max-hours {args.max_hours} reached.")
            break
        if r["kind"] != "dummy" and not args.dry_run:
            free = free_mb(args.gpu_list)
            if free is None:
                sys.exit("nvidia-smi not available; cannot check the GPUs.")
            low = {g: f for g, f in free.items() if f < args.min_free_mb}
            if low:
                sys.exit(f"Stopping: not enough free GPU memory {low} (MB).")

        try:
            script = make_script(args, r, args.port + i % 1000)
        except RuntimeError as e:
            print(f"skip  {r['id']}: {e}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        (out / "run.sh").write_text(script)
        if args.dry_run:
            print(f"wrote {out / 'run.sh'}")
            continue

        info = {"id": r["id"], "kind": r["kind"], "script": r["script"], "overrides": r["overrides"],
                "data": args.data, "upstream_commit": UPSTREAM_COMMIT, "repo_commit": repo_commit(),
                "host": socket.gethostname(), "gpus": gpu_info(args.gpu_list),
                "started": dt.datetime.now().isoformat(timespec="seconds")}
        print(f"start {r['id']}", flush=True)
        t0 = time.time()
        with open(out / "run.log", "w") as log:
            code = subprocess.run(["bash", str(out / "run.sh")], cwd=UPSTREAM,
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        info.update(exit_code=code, minutes=round((time.time() - t0) / 60, 1),
                    finished=dt.datetime.now().isoformat(timespec="seconds"))
        metrics = parse_metrics(r["kind"], (out / "run.log").read_text(errors="replace"))
        (out / "run.json").write_text(json.dumps(info, indent=1))
        if code == 0 and metrics:
            (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
            print(f"done  {r['id']}  {info['minutes']} min  MAPE {metrics['test_MAPE']:.3f}", flush=True)
        else:
            print(f"FAIL  {r['id']}  exit {code}; see {out / 'run.log'}", flush=True)


if __name__ == "__main__":
    main()
