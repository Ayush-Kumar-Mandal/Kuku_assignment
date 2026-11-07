"""Everything the Streamlit app does besides drawing: list/load runs, launch/resume/stop runs,
read live progress. Kept free of Streamlit so it can be unit-tested.

Runs are launched as a separate `python run.py ...` process (the same CLI used without the
UI), not inside the Streamlit script: a full run takes about an hour, and a run inside the
script would die on a browser refresh. The UI shows progress from the run's checkpoint and
the files run.py exports. Stopping kills the process; the checkpoint keeps every finished
step, so Resume continues from there.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from config.loader import ROOT, Settings

RUN_PY = ROOT / "run.py"


# ------------------------------------------------------------------------------ paths
def runs_dir(settings: Settings) -> Path:
    return settings.resolve(settings.paths.output_dir) / "runs"


def run_dir(settings: Settings, run_id: str) -> Path:
    return runs_dir(settings) / run_id


def log_path(settings: Settings, run_id: str) -> Path:
    return settings.resolve(settings.paths.logs_dir) / "runs" / f"{run_id}.log"


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


# ----------------------------------------------------------------------------- choices
def model_choices(settings: Settings) -> list[str]:
    """Every provider:model named anywhere in the config (main or backup): no hardcoded list."""
    return sorted({f"{c.provider}:{c.model}" for spec in settings.models.values() for c in spec.chain()})


def upload_types(parser_glob: str) -> list[str] | None:
    """File extensions for the uploader, from parser.yaml's file_glob (e.g. '*.txt' -> ['txt'])."""
    m = re.fullmatch(r"\*\.(\w+)", parser_glob.strip())
    return [m.group(1)] if m else None


def save_uploads(files, dest: Path) -> Path:
    """Write uploaded files to dest, keeping only each file's base name (no path tricks)."""
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = Path(str(f.name).replace("\\", "/")).name
        if name:
            (dest / name).write_bytes(f.getvalue())
    return dest


# ---------------------------------------------------------------------------- process
def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) would TERMINATE the process on Windows; ask for its exit code instead.
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259               # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


# The run's OS lock (graph.runner.holder) is the real "is it running" signal and can't go stale.
# The pid in launch.json only covers the moment between starting run.py and run.py taking that
# lock. After that it is not trusted: Windows reuses process ids, and an old run's pid once
# belonged to the Streamlit app itself (the Review tab then said "running" forever, and Stop
# would have killed the app).
LAUNCH_GRACE_S = 120


def _launch_pid(settings: Settings, run_id: str) -> int | None:
    """The pid the UI started, only while the launch is recent enough to still be that process."""
    info = _read_json(run_dir(settings, run_id) / "launch.json", {})
    try:
        started = time.mktime(time.strptime(info.get("started_at", ""), "%Y-%m-%d %H:%M:%S"))
    except (TypeError, ValueError):
        return None
    return info.get("pid") if 0 <= time.time() - started <= LAUNCH_GRACE_S else None


def is_running(settings: Settings, run_id: str) -> bool:
    """A process is driving the run: one holding the run's lock (UI launch or CLI resume), or a
    run the UI started moments ago that hasn't taken the lock yet."""
    from graph.runner import holder
    return holder(settings, run_id) is not None or pid_alive(_launch_pid(settings, run_id))


def _spawn(settings: Settings, run_id: str, args: list[str]) -> subprocess.Popen:
    """Raises graph.runner.RunLocked if another process is already running this run (run.py
    would refuse anyway; checking here keeps launch.json pointing at the live process)."""
    from graph.runner import RunLocked, holder
    owner = holder(settings, run_id)
    if owner is not None:
        raise RunLocked(run_id, owner)
    log = log_path(settings, run_id)
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(RUN_PY), *args, "-v"]
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(args)}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT, creationflags=flags,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    d = run_dir(settings, run_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "launch.json").write_text(json.dumps({"pid": proc.pid, "args": cmd, "log": str(log),
                                               "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1),
                                   encoding="utf-8")
    return proc


def launch(settings_path: str | Path, run_id: str, input_dir: str | Path, show: str, genre: str,
           agents: str = "llm", review: bool = True, memory: bool = True, joiner: bool = True,
           episodes: str | None = None, models: dict[str, str] | None = None) -> subprocess.Popen:
    from config.loader import load_settings
    settings = load_settings(settings_path)
    args = ["--input", str(input_dir), "--show", show, "--genre", genre, "--config", str(settings_path),
            "--run-id", run_id, "--agents", agents]
    args += [] if review else ["--no-review"]
    args += [] if memory else ["--no-memory"]
    args += [] if joiner else ["--no-joiner"]
    args += ["--episodes", episodes] if episodes else []
    for agent, value in (models or {}).items():
        args += ["--model", f"{agent}={value}"]
    return _spawn(settings, run_id, args)


def resume(settings_path: str | Path, run_id: str, decisions: str | Path | None = None) -> subprocess.Popen:
    """Continue a stopped or paused run. Agents and model choices come from the run itself."""
    from config.loader import load_settings
    args = ["--resume", run_id, "--config", str(settings_path)]
    args += ["--decisions", str(decisions)] if decisions else []
    return _spawn(load_settings(settings_path), run_id, args)


def stop(settings: Settings, run_id: str) -> bool:
    from graph.runner import holder
    pid = (holder(settings, run_id) or {}).get("pid")           # the process running it (UI or CLI)
    if not pid_alive(pid):
        pid = _launch_pid(settings, run_id)                     # just launched, no lock yet
    if not pid_alive(pid):
        return False
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        os.kill(int(pid), 15)
    return True


# ------------------------------------------------------------------------------ read
def list_runs(settings: Settings) -> list[dict]:
    """Runs with an output folder, newest first: id, state (running if its process is alive)."""
    out = []
    root = runs_dir(settings)
    for d in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if root.exists() else []:
        summary = _read_json(d / "run_summary.json")
        launch_info = _read_json(d / "launch.json", {})
        if summary is None and not launch_info:
            continue
        running = is_running(settings, d.name)
        state = "running" if running else (summary or {}).get("state", "starting")
        out.append({"run_id": d.name, "state": state, "show": (summary or {}).get("show"),
                    "written_at": (summary or {}).get("written_at"), "running": running})
    return out


def load_run(settings: Settings, run_id: str) -> dict:
    d = run_dir(settings, run_id)
    return {"dir": d, "summary": _read_json(d / "run_summary.json"), "promos": _read_json(d / "promos.json", []),
            "candidates": _read_json(d / "raw" / "candidates.json", []),
            "launch": _read_json(d / "launch.json", {}),
            "files": {name: d / rel for name, rel in [("promos.csv", "promos.csv"), ("promos.json", "promos.json"),
                                                      ("candidates.csv", "raw/candidates.csv"),
                                                      ("candidates.json", "raw/candidates.json"),
                                                      ("report.md", "report.md")] if (d / rel).exists()}}


def load_trace(settings: Settings, run_id: str, promo_id: str) -> dict | None:
    return _read_json(run_dir(settings, run_id) / "traces" / f"{promo_id}.json")


def progress(values: dict, next_nodes: list[str]) -> dict:
    """Where a run is, from its checkpointed state (works while run.py is still writing)."""
    total = len(values.get("episodes") or []) or 1
    read, cut = values.get("reader_cursor", 0), values.get("cutter_cursor", 0)
    cands = values.get("candidates") or {}
    evaluated = sum(1 for p in cands if (values.get("evaluations") or {}).get(p))
    frac = 0.35 * read / total + 0.35 * cut / total + 0.05 * bool(values.get("joiner_done")) \
        + 0.25 * (evaluated / len(cands) if cands else 0)
    return {"stage": next_nodes[0] if next_nodes else "done", "total": total, "read": read, "cut": cut,
            "candidates": len(cands), "evaluated": evaluated, "pending_eval": len(values.get("pending_eval") or []),
            "ranked": len(values.get("promos") or []), "errors": len(values.get("errors") or []),
            "fraction": round(min(frac, 1.0), 3)}


def live(settings: Settings, run_id: str) -> dict:
    """Checkpoint state + whether the run's process is alive."""
    from graph.build import build_graph
    from graph.runner import open_checkpointer, status
    try:
        with open_checkpointer(settings) as cp:
            st = status(build_graph(cp), run_id, settings)
    except Exception as e:  # DB briefly locked by the writer, or not created yet
        return {"state": "unknown", "error": str(e), "running": False, "progress": None, "values": {}}
    running = is_running(settings, run_id)
    return {"state": "running" if running else st["state"], "checkpoint_state": st["state"], "running": running,
            "progress": progress(st["values"], st["next"]) if st["values"] else None, "values": st["values"],
            "interrupt": st["interrupt"]}


def log_tail(settings: Settings, run_id: str, lines: int = 25) -> str:
    p = log_path(settings, run_id)
    if not p.exists():
        return ""
    keep = [l for l in p.read_text(encoding="utf-8", errors="replace").splitlines()
            if "HTTP Request:" not in l and "_base_client" not in l]
    return "\n".join(keep[-lines:])


# ------------------------------------------------------------------------- evaluation
# The Evaluation page only READS what the evaluate/ CLIs write (labels, compare, agreement);
# making labels or reports stays on the command line, where the blind labelling happens.
def _eval_base(settings: Settings) -> Path:
    return settings.resolve(settings.evaluation.dir)


def episode_ranges(eps: list[int]) -> str:
    """[1, 2, 3, 5] -> '1-3, 5'."""
    out, eps = [], sorted(set(eps))
    for e in eps:
        if out and e == out[-1][1] + 1:
            out[-1][1] = e
        else:
            out.append([e, e])
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in out)


def list_label_files(settings: Settings) -> list[dict]:
    """Every labels file, with who made it (human / ai / planted) and how much it covers."""
    from evaluate.labels import load_labels
    rows = []
    for p in sorted((_eval_base(settings) / "labels").glob("*.yaml")):
        try:
            lab = load_labels(p)
        except Exception:  # a half-edited or broken file: skip it, the CLI will report it
            continue
        rows.append({"file": p.name, "show": lab.show, "source": lab.label_source, "labeled_by": lab.labeled_by,
                     "models": ", ".join(lab.labeler_models), "episodes": episode_ranges(lab.episodes),
                     "moments": len(lab.moments), "locked": lab.locked_at or "", "path": p})
    return rows


def _newest_first(paths) -> list[Path]:
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def list_agreements(settings: Settings) -> list[dict]:
    """evaluate.agreement reports (<candidate>__vs__<reference>.json), newest first."""
    out = []
    for p in _newest_first((_eval_base(settings) / "agreement").glob("*.json")):
        r = _read_json(p)
        if isinstance(r, dict) and "verdict" in r:
            out.append({"name": p.stem, "report": r, "md": p.with_suffix(".md")})
    return out


def list_run_evaluations(settings: Settings) -> list[dict]:
    """evaluate.compare reports (<eval dir>/<run_id>/evaluation.json), newest first."""
    out = []
    for p in _newest_first(_eval_base(settings).glob("*/evaluation.json")):
        r = _read_json(p)
        if isinstance(r, dict) and "recall" in r:
            out.append({"run_id": p.parent.name, "report": r, "md": p.with_suffix(".md")})
    return out
