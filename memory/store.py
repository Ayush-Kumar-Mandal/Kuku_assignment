"""MemoryStore: the editor's long-term memory (spec section 8), SQLite at settings.memory.db_path.

Why our own SQLite and not LangGraph's Store: the installed langgraph.store.sqlite.SqliteStore
is a namespaced key-value store (put/get/search with equality filters). The spec needs an
append-only log, top-N rules ORDER BY confidence with scope filters, atomic support /
contradiction counters, conflict links and a version history: plain SQL fits that cleanly,
the KV store doesn't. Agents and nodes only talk to this class.

Tables
  feedback_log   append-only (UPDATE/DELETE blocked by triggers): every editor decision
  guidelines     rule text, scope (global/genre/show), counts, confidence, status
                 (active / retired / needs_review), source feedback ids
  examples       approved / rejected promos with the editor's reason and evaluator scores
  evidence       (rule, feedback, supports|contradicts): each feedback counts once per rule
  conflicts      pairs of rules that contradict each other (both set to needs_review)
  versions       v1, v2...: a snapshot of the active rules each time guidelines change
  memory_writes  which runs were written (memory_write is idempotent across resumes)
Confidence = (supporting + 1) / (supporting + contradicting + 2). A rule whose contradictions
outweigh its support goes to needs_review; nothing is ever deleted automatically.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

STATUSES = {"active", "retired", "needs_review"}
SCOPES = {"global", "genre", "show"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback_log (
  id INTEGER PRIMARY KEY, show TEXT, genre TEXT, promo_id TEXT, promo TEXT, moment_type TEXT,
  decision TEXT, comment TEXT, fix_interpretation TEXT, timestamp TEXT, guidelines_version TEXT,
  run_id TEXT, round INTEGER, UNIQUE (run_id, promo_id, round, decision));
CREATE TRIGGER IF NOT EXISTS feedback_log_no_update BEFORE UPDATE ON feedback_log
  BEGIN SELECT RAISE(ABORT, 'feedback_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS feedback_log_no_delete BEFORE DELETE ON feedback_log
  BEGIN SELECT RAISE(ABORT, 'feedback_log is append-only'); END;
CREATE TABLE IF NOT EXISTS guidelines (
  id INTEGER PRIMARY KEY, text TEXT NOT NULL, scope TEXT NOT NULL, scope_value TEXT,
  supporting_count INTEGER DEFAULT 0, contradicting_count INTEGER DEFAULT 0, confidence REAL DEFAULT 0.5,
  status TEXT NOT NULL, created_at TEXT, updated_at TEXT, source_feedback_ids TEXT DEFAULT '[]',
  created_by TEXT DEFAULT 'ai');
CREATE TABLE IF NOT EXISTS examples (
  id INTEGER PRIMARY KEY, promo_id TEXT, promo TEXT, label TEXT, reason TEXT, show TEXT, genre TEXT, moment_type TEXT,
  scores TEXT, final REAL, run_id TEXT, created_at TEXT, UNIQUE (run_id, promo_id));
CREATE TABLE IF NOT EXISTS evidence (
  rule_id INTEGER, feedback_id INTEGER, relation TEXT, created_at TEXT, PRIMARY KEY (rule_id, feedback_id));
CREATE TABLE IF NOT EXISTS conflicts (rule_a INTEGER, rule_b INTEGER, why TEXT, created_at TEXT,
  PRIMARY KEY (rule_a, rule_b));
CREATE TABLE IF NOT EXISTS versions (version INTEGER PRIMARY KEY, created_at TEXT, reason TEXT, active_rules TEXT);
CREATE TABLE IF NOT EXISTS memory_writes (run_id TEXT PRIMARY KEY, stage TEXT, updated_at TEXT);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def confidence(supporting: int, contradicting: int) -> float:
    return round((supporting + 1) / (supporting + contradicting + 2), 3)


def scope_for(feedback_rows: list[dict]) -> tuple[str, str | None]:
    """A rule seen with one show is that show's; across shows of one genre, the genre's;
    across genres, global (spec: global if it holds across shows, else genre or show)."""
    shows = {(r.get("show") or "").strip().lower() for r in feedback_rows}
    genres = {(r.get("genre") or "").strip().lower() for r in feedback_rows}
    if len(shows) <= 1:
        return "show", (feedback_rows[0].get("show") if feedback_rows else None)
    if len(genres) == 1:
        return "genre", feedback_rows[0].get("genre")
    return "global", None


def example_summary(promo: dict, label: str, reason: str) -> str:
    segs = " + ".join(f"ep{s['episode']} {s['start_timestamp']}-{s['end_timestamp']}" for s in promo.get("segments") or [])
    text = (f"{promo.get('moment_type', '')} promo ({promo.get('total_duration_seconds', '?')}s{', ' + segs if segs else ''}): "
            f"opens on \"{promo.get('hook', '')}\", ends on \"{promo.get('cliffhanger', '')}\"")
    return text + (f". Editor's reason: {reason}" if reason else "")


class MemoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Streamlit and run.py processes share the file: WAL lets readers work during writes.
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ---------------------------------------------------------------- feedback_log
    def log_feedback(self, *, show: str, genre: str, promo_id: str, promo: dict, moment_type: str | None,
                     decision: str, comment: str, guidelines_version: str | None, run_id: str, round: int,
                     fix_interpretation: str | None = None) -> int:
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO feedback_log (show, genre, promo_id, promo, moment_type, decision, comment, "
                "fix_interpretation, timestamp, guidelines_version, run_id, round) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (show, genre, promo_id, json.dumps(promo, ensure_ascii=False), moment_type, decision, comment or "",
                 fix_interpretation, _now(), guidelines_version, run_id, round))
        return self.conn.execute("SELECT id FROM feedback_log WHERE run_id=? AND promo_id=? AND round=? AND decision=?",
                                 (run_id, promo_id, round, decision)).fetchone()["id"]

    def feedback(self, ids: list[int] | None = None, limit: int | None = None) -> list[dict]:
        q, args = "SELECT * FROM feedback_log", []
        if ids is not None:
            q += f" WHERE id IN ({','.join('?' * len(ids))})" if ids else " WHERE 0"
            args = list(ids)
        q += " ORDER BY id" + (f" LIMIT {int(limit)}" if limit else "")
        return [self._fb(r) for r in self.conn.execute(q, args)]

    def unused_feedback(self, limit: int) -> list[dict]:
        """Feedback with a comment that no rule is built on yet (newest first): material for new rules."""
        used = {i for r in self.rules() for i in r["source_feedback_ids"]}
        rows = self.conn.execute("SELECT * FROM feedback_log WHERE comment <> '' ORDER BY id DESC").fetchall()
        return [self._fb(r) for r in rows if r["id"] not in used][:limit]

    @staticmethod
    def _fb(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["promo"] = json.loads(d["promo"] or "{}")
        return d

    # -------------------------------------------------------------------- examples
    def add_example(self, *, promo: dict, label: str, reason: str, show: str, genre: str, moment_type: str | None,
                    scores: dict | None, run_id: str) -> int:
        if label not in ("approved", "rejected"):
            raise ValueError(f"example label must be approved or rejected, not {label!r}")
        with self.conn:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO examples (promo_id, promo, label, reason, show, genre, moment_type, scores, final, "
                "run_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (promo.get("promo_id"), json.dumps(promo, ensure_ascii=False), label, reason or "", show, genre, moment_type,
                 json.dumps(scores or {}), (scores or {}).get("final"), run_id, _now()))
        return cur.lastrowid

    def examples(self, genre: str | None = None, label: str | None = None, limit: int | None = None,
                 exclude: set[int] | None = None) -> list[dict]:
        q, args = "SELECT * FROM examples WHERE 1", []
        if genre is not None:
            q += " AND lower(genre) = lower(?)"
            args.append(genre)
        if label:
            q += " AND label = ?"
            args.append(label)
        q += " ORDER BY final DESC, id"
        out = []
        for r in self.conn.execute(q, args):
            if exclude and r["id"] in exclude:
                continue
            d = dict(r)
            d["promo"], d["scores"] = json.loads(d["promo"]), json.loads(d["scores"] or "{}")
            out.append(d)
            if limit and len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------------ guidelines
    def _row(self, r: sqlite3.Row | None) -> dict | None:
        if r is None:
            return None
        d = dict(r)
        d["source_feedback_ids"] = json.loads(d["source_feedback_ids"] or "[]")
        d["conflicts_with"] = [x["other"] for x in self.conn.execute(
            "SELECT rule_b AS other FROM conflicts WHERE rule_a=? UNION SELECT rule_a FROM conflicts WHERE rule_b=?",
            (d["id"], d["id"]))]
        return d

    def rule(self, rule_id: int) -> dict | None:
        return self._row(self.conn.execute("SELECT * FROM guidelines WHERE id=?", (rule_id,)).fetchone())

    def rules(self, status: str | set[str] | None = None) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM guidelines ORDER BY confidence DESC, supporting_count DESC, id").fetchall()
        want = {status} if isinstance(status, str) else status
        return [self._row(r) for r in rows if not want or r["status"] in want]

    def _need(self, rule_id: int) -> dict:
        r = self.rule(rule_id)
        if r is None:
            raise ValueError(f"no rule {rule_id}")
        return r

    def add_rule(self, text: str, scope: str, source_feedback_ids: list[int], status: str = "needs_review",
                 scope_value: str | None = None, created_by: str = "ai") -> int:
        if scope not in SCOPES or status not in STATUSES or not text.strip():
            raise ValueError(f"bad rule: scope={scope} status={status} text={text!r}")
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO guidelines (text, scope, scope_value, status, created_at, updated_at, source_feedback_ids, "
                "created_by, confidence) VALUES (?,?,?,?,?,?,?,?,?)",
                (text.strip(), scope, scope_value, status, _now(), _now(), json.dumps(sorted(set(source_feedback_ids))),
                 created_by, confidence(0, 0)))
            self._bump(f"rule {cur.lastrowid} added ({status}) by {created_by}")
        return cur.lastrowid

    def edit_rule(self, rule_id: int, text: str, by: str = "editor"):
        self._need(rule_id)
        with self.conn:
            self.conn.execute("UPDATE guidelines SET text=?, updated_at=? WHERE id=?", (text.strip(), _now(), rule_id))
            self._bump(f"rule {rule_id} edited by {by}")

    def set_status(self, rule_id: int, status: str, by: str = "editor"):
        if status not in STATUSES:
            raise ValueError(f"status must be one of {sorted(STATUSES)}")
        self._need(rule_id)
        with self.conn:
            self.conn.execute("UPDATE guidelines SET status=?, updated_at=? WHERE id=?", (status, _now(), rule_id))
            self._bump(f"rule {rule_id} -> {status} by {by}")

    def delete_rule(self, rule_id: int, by: str = "editor"):
        """Editor-only: the AI never deletes rules."""
        self._need(rule_id)
        with self.conn:
            for sql in ("DELETE FROM guidelines WHERE id=?", "DELETE FROM evidence WHERE rule_id=?",
                        "DELETE FROM conflicts WHERE rule_a=? OR rule_b=?"):
                self.conn.execute(sql, (rule_id,) * sql.count("?"))
            self._bump(f"rule {rule_id} deleted by {by}")

    def add_evidence(self, rule_id: int, feedback_id: int, relation: str) -> bool:
        """Count one feedback item for/against a rule (once). Contradictions outweighing support
        flag the rule for the editor (needs_review); it is never deleted automatically."""
        if relation not in ("supports", "contradicts"):
            raise ValueError(relation)
        r = self._need(rule_id)
        with self.conn:
            cur = self.conn.execute("INSERT OR IGNORE INTO evidence VALUES (?,?,?,?)", (rule_id, feedback_id, relation, _now()))
            if not cur.rowcount:
                return False
            s, c = (self.conn.execute("SELECT COUNT(*) FROM evidence WHERE rule_id=? AND relation=?", (rule_id, rel)).fetchone()[0]
                    for rel in ("supports", "contradicts"))
            status = "needs_review" if c > s and r["status"] == "active" else r["status"]
            self.conn.execute("UPDATE guidelines SET supporting_count=?, contradicting_count=?, confidence=?, status=?, "
                              "updated_at=? WHERE id=?", (s, c, confidence(s, c), status, _now(), rule_id))
            if status != r["status"]:
                self._bump(f"rule {rule_id} -> needs_review (contradicted {c} vs supported {s})")
        return True

    def mark_conflict(self, a: int, b: int, why: str = ""):
        self._need(a), self._need(b)
        lo, hi = sorted((a, b))
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO conflicts VALUES (?,?,?,?)", (lo, hi, why, _now()))
            self.conn.execute("UPDATE guidelines SET status='needs_review', updated_at=? WHERE id IN (?,?) AND status<>'retired'",
                              (_now(), a, b))
            self._bump(f"rules {a} and {b} conflict: both need review")

    # -------------------------------------------------------------------- versions
    def version(self) -> int:
        return self.conn.execute("SELECT COALESCE(MAX(version), 0) FROM versions").fetchone()[0]

    def version_label(self) -> str:
        return f"v{self.version()}"

    def _bump(self, reason: str):
        """New guidelines version (inside the caller's transaction) with the active rules then."""
        active = [{"id": r["id"], "text": r["text"], "scope": r["scope"], "scope_value": r["scope_value"]}
                  for r in self.conn.execute("SELECT * FROM guidelines WHERE status='active' ORDER BY id")]
        self.conn.execute("INSERT INTO versions VALUES (?,?,?,?)",
                          (self.version() + 1, _now(), reason, json.dumps(active, ensure_ascii=False)))

    def history(self) -> list[dict]:
        return [{**dict(r), "active_rules": json.loads(r["active_rules"])}
                for r in self.conn.execute("SELECT * FROM versions ORDER BY version DESC")]

    # --------------------------------------------------------------------- reading
    def for_run(self, show: str, genre: str, top_rules: int, approved: int, rejected: int) -> dict:
        """What memory_read injects: active rules for this show/genre/global (top N by
        confidence) + approved/rejected examples from the same genre, then any genre."""
        sh, ge = (show or "").strip().lower(), (genre or "").strip().lower()
        rules = [r for r in self.rules("active")
                 if r["scope"] == "global"
                 or (r["scope"] == "genre" and (r["scope_value"] or "").strip().lower() == ge)
                 or (r["scope"] == "show" and (r["scope_value"] or "").strip().lower() == sh)][:top_rules]
        examples = []
        for label, n in (("approved", approved), ("rejected", rejected)):
            got = self.examples(genre=genre, label=label, limit=n) if n else []
            if len(got) < n:                                     # fall back to any genre
                got += self.examples(label=label, limit=n - len(got), exclude={e["id"] for e in got})
            examples += got
        return {
            "guidelines_version": self.version_label(),
            "rules": [{"id": r["id"], "text": r["text"], "scope": r["scope"], "scope_value": r["scope_value"],
                       "confidence": r["confidence"]} for r in rules],
            "examples": [{"id": e["id"], "label": e["label"], "promo_id": e["promo"].get("promo_id"),
                          "moment_type": e["moment_type"], "genre": e["genre"], "final": e["final"],
                          "reason": e["reason"], "summary": example_summary(e["promo"], e["label"], e["reason"])}
                         for e in examples],
        }

    # ------------------------------------------------------------- write markers
    def write_stage(self, run_id: str) -> str | None:
        r = self.conn.execute("SELECT stage FROM memory_writes WHERE run_id=?", (run_id,)).fetchone()
        return r["stage"] if r else None

    def mark_written(self, run_id: str, stage: str):
        with self.conn:
            self.conn.execute("INSERT INTO memory_writes VALUES (?,?,?) ON CONFLICT(run_id) DO UPDATE SET stage=?, updated_at=?",
                              (run_id, stage, _now(), stage, _now()))

    def stats(self) -> dict[str, Any]:
        q = lambda sql: self.conn.execute(sql).fetchone()[0]
        return {"version": self.version_label(), "feedback": q("SELECT COUNT(*) FROM feedback_log"),
                "examples": q("SELECT COUNT(*) FROM examples"),
                **{f"rules_{s}": q(f"SELECT COUNT(*) FROM guidelines WHERE status='{s}'") for s in sorted(STATUSES)}}
