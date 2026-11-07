"""Start, resume and inspect graph runs. Used by the CLI (Step 6) and Streamlit (Step 7).

A run is one LangGraph thread (thread_id = run_id) in the SQLite checkpoint DB:
  start   : run(graph, ctx, run_id, initial_state(...))
  crash / rate-limit stop, then resume : run(graph, ctx, run_id)            (input None)
  editor decisions                     : run(graph, ctx, run_id, Command(resume=decisions))

Only one process may drive a run at a time: run.py holds run_lock(settings, run_id) for the
whole start/resume (see run_lock).
"""
from __future__ import annotations

import json
import os
import socket
import sqlite3
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from config.loader import Settings


@contextmanager
def open_checkpointer(settings: Settings, path: str | Path | None = None):
    db = Path(path) if path else settings.resolve(settings.paths.checkpoint_db)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db, check_same_thread=False)   # Streamlit reruns on other threads
    try:
        yield SqliteSaver(conn)
    finally:
        conn.close()


# ------------------------------------------------------------------------------ run lock
# An OS file lock (msvcrt on Windows, flock elsewhere) on <checkpoint dir>/locks/<run_id>.lock.
# The OS drops it when the holder exits for any reason (finish, crash, Ctrl+C, taskkill), so a
# lock can never go stale and there is nothing to clean up. The lock file is never deleted:
# deleting it would let a second process lock a fresh file while the first still holds the old one.
# While held, the file holds the owner's pid / start time / command (bytes 0..n, readable);
# the locked byte sits far past that, at _LOCK_OFFSET, because Windows can't read locked bytes.
_LOCK_OFFSET = 1 << 20


class RunLocked(RuntimeError):
    """Another process is already running this run."""

    def __init__(self, run_id: str, owner: dict):
        self.run_id, self.owner = run_id, owner
        who = ", ".join(f"{k} {owner[k]}" for k in ("pid", "host", "started_at") if owner.get(k)) or "owner unknown"
        cmd = f"\n  its command: {owner['command']}" if owner.get("command") else ""
        super().__init__(f"run '{run_id}' is already running in another process ({who}).{cmd}\n"
                         f"Wait for it to finish, or stop it first (Stop run in the UI, or kill that pid).")


def lock_path(settings: Settings, run_id: str) -> Path:
    return settings.resolve(settings.paths.checkpoint_db).parent / "locks" / f"{run_id}.lock"


def _try_lock(fd: int) -> bool:
    os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(fd: int) -> None:
    os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)


def _read_owner(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, ValueError):
        return {}


@contextmanager
def run_lock(settings: Settings, run_id: str, wait: float = 1.5):
    """Hold the run's lock, or raise RunLocked naming the process that has it.
    `wait`: if the lock is taken but no owner is written yet (an owner that is just starting,
    or a holder() probe from the UI), retry for this long before giving up."""
    path = lock_path(settings, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0))
    try:
        deadline = time.monotonic() + wait
        while not _try_lock(fd):
            owner = _read_owner(path)
            if owner or time.monotonic() >= deadline:
                raise RunLocked(run_id, owner)
            time.sleep(0.1)
        owner = {"pid": os.getpid(), "host": socket.gethostname(), "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "command": " ".join([Path(sys.executable).name, *sys.argv])}
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(owner).encode("utf-8"))
        try:
            yield owner
        finally:
            os.ftruncate(fd, 0)                  # still holding the lock: no one is mid-read of a new owner
            _unlock(fd)
    finally:
        os.close(fd)


def holder(settings: Settings, run_id: str) -> dict | None:
    """The owner of the run's lock ({} if it hasn't written itself yet), or None if no process
    is running it. Probes without blocking; never creates the lock file."""
    path = lock_path(settings, run_id)
    try:
        fd = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
    except FileNotFoundError:
        return None
    try:
        if _try_lock(fd):
            _unlock(fd)
            return None
        return _read_owner(path)
    finally:
        os.close(fd)


def new_run_id(show: str) -> str:
    slug = "".join(c if c.isalnum() else "-" for c in show.lower()).strip("-")[:30] or "show"
    return f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}"


def initial_state(run_id: str, show: str, genre: str, input_dir: str | Path, settings: Settings,
                  memory: bool | None = None, joiner: bool | None = None, editor_review: bool | None = None,
                  episodes: list[int] | None = None, agents: str = "llm",
                  model_overrides: dict[str, str] | None = None) -> dict:
    sw = settings.switches
    return {
        "run_id": run_id, "show": show, "genre": genre, "input_dir": str(input_dir),
        "episode_filter": sorted(set(episodes)) if episodes else None,
        "agents_kind": agents,              # llm | stub; a resume reuses it
        "model_overrides": dict(model_overrides or {}),   # --model / UI choices; a resume reuses them
        "switches": {
            "memory": sw.memory if memory is None else memory,
            "joiner": sw.joiner if joiner is None else joiner,
            "editor_review": sw.editor_review if editor_review is None else editor_review,
        },
    }


def config_for(run_id: str, settings: Settings) -> dict:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": settings.graph.recursion_limit}


def run(graph, ctx, run_id: str, inputs: dict | Command | None = None) -> dict[str, Any]:
    """Run until the graph ends or pauses. Returns the final state values; if paused at
    editor_review, the payload is under '__interrupt__'."""
    return graph.invoke(inputs, config_for(run_id, ctx.settings), context=ctx,
                        durability=ctx.settings.graph.durability)


def status(graph, run_id: str, settings: Settings) -> dict:
    """Where a run stands: done, paused for review, or stopped mid-way (resumable)."""
    snap = graph.get_state(config_for(run_id, settings))
    interrupts = [i.value for t in snap.tasks for i in (t.interrupts or ())]
    if not snap.values:
        state = "not_started"
    elif interrupts:
        state = "awaiting_review"
    elif snap.next:
        state = "stopped"          # crashed or rate-limited: run(..., inputs=None) resumes
    else:
        state = "done"
    return {"state": state, "next": list(snap.next), "interrupt": interrupts[0] if interrupts else None,
            "values": snap.values}


def build_context(settings: Settings, run_id: str, show: str, genre: str, agents: str = "llm",
                  llm_parser_fallback: bool = False):
    """RunContext for a run. agents="llm" uses the real agents, "stub" the deterministic ones.
    Built fresh on every start/resume; it is never checkpointed."""
    from config.loader import load_moment_types, load_parser_config
    from graph.context import RunContext

    pcfg, mtypes = load_parser_config(settings=settings), load_moment_types(settings=settings)
    if agents == "stub":
        # Dry runs never read or write the editor's memory: fake decisions must not teach it.
        from agents.stubs import StubAgents
        return RunContext(settings=settings, parser_cfg=pcfg, moment_types=mtypes, agents=StubAgents(settings))
    from agents.llm_agents import LLMAgents
    from llm.client import LLMClient
    from llm.parser_fallback import make_parser_fallback

    from memory.store import MemoryStore
    client = LLMClient(settings, run_id=run_id)
    return RunContext(settings=settings, parser_cfg=pcfg, moment_types=mtypes, client=client,
                      memory=MemoryStore(settings.resolve(settings.memory.db_path)),
                      agents=LLMAgents(client, settings, pcfg, mtypes, show=show, genre=genre),
                      parser_fallback=make_parser_fallback(client, pcfg) if llm_parser_fallback else None)
