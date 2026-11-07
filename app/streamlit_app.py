"""Promo Finder UI.   streamlit run app/streamlit_app.py

New run: upload screenplays (or point at a folder), show name + genre, models, switches,
start. Runs execute as a separate `run.py` process (see app/data.py), so they survive a
browser refresh; this page shows their live progress and the files they write.
Per run: Overview (progress, usage, log), Promos (ranked list, downloads, details),
Trace (any promo, ranked or dropped), Review (Phase 2).
Evaluation: the evaluate/ reports (tool vs labels, labeller agreement, label files); read-only.
Memory: the editor's learned rules (Phase 3).

PROMO_SETTINGS=<path> selects another settings file (used by tests).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app import data, review, ui
from config.loader import load_parser_config, load_settings
from graph.export import segs_text
from graph.runner import RunLocked, new_run_id

SETTINGS_PATH = Path(os.environ.get("PROMO_SETTINGS", ROOT / "config" / "settings.yaml"))
AGENT_LABELS = {"reader": "Reader", "cutter": "Cutter", "joiner": "Joiner", "evaluator": "Evaluator"}
PAGES = [("new", "New run", ":material/add_circle:"), ("runs", "Runs", ":material/movie:"),
         ("evaluation", "Evaluation", ":material/insights:"), ("memory", "Memory", ":material/psychology:")]
STAGES = {"parse": "Reading the files", "memory_read": "Loading memory", "reader": "Reading episodes",
          "cutter": "Cutting promos", "joiner": "Joining clips", "evaluator": "Scoring promos", "rank": "Ranking",
          "editor_review": "Waiting for your review", "feedback": "Applying your fixes",
          "memory_write": "Saving memory", "done": "Finished"}

st.set_page_config(page_title="Promo Finder", page_icon=":material/movie_filter:", layout="wide")
settings = load_settings(SETTINGS_PATH)


# ----------------------------------------------------------------------------- sidebar
def sidebar() -> tuple[str, str | None]:
    """Menu (one button per page, the current one highlighted) and, on Runs, the run picker."""
    runs = data.list_runs(settings)
    ids = [r["run_id"] for r in runs]
    states = {r["run_id"]: r["state"] for r in runs}
    if st.session_state.get("goto") in ids:          # set after starting a run
        st.session_state["run_select"] = st.session_state.pop("goto")
        st.session_state["page"] = "runs"
    st.session_state.setdefault("page", "runs" if ids else "new")
    sb = st.sidebar
    sb.markdown("### :material/movie_filter: Promo Finder")
    sb.caption("Finds 20-60 s promo clips in microdrama screenplays.")
    for key, label, icon in PAGES:
        if sb.button(label, key=f"nav_{key}", icon=icon, width="stretch",
                     type="primary" if st.session_state["page"] == key else "tertiary"):
            st.session_state["page"] = key
            st.rerun()
    page, choice = st.session_state["page"], None
    if page == "runs":
        sb.divider()
        if not ids:
            sb.caption("No runs yet. Start one from **New run**.")
        else:
            if st.session_state.get("run_select") not in ids:
                st.session_state["run_select"] = ids[0]
            choice = sb.selectbox("Run", ids, key="run_select",
                                  format_func=lambda i: f"{ui.state_dot(states.get(i))} {i}")
            sb.caption(" · ".join(f"{d} {lbl.lower()}" for d, lbl in
                                  [("🔵", "running"), ("🟠", "review"), ("🟢", "done"), ("⚪", "stopped")]))
    sb.divider()
    sb.caption(f":material/lock: Local only · settings `{SETTINGS_PATH.name}`")
    return page, choice


# ----------------------------------------------------------------------------- new run
def new_run_page():
    ui.page_header("New run", ":material/add_circle:",
                   "Give the show, add the screenplays, pick the options and start. The run keeps going in the "
                   "background, even if you close this page.")
    pcfg = load_parser_config(settings=settings)
    choices = data.model_choices(settings)
    with st.form("new_run", border=False):
        with st.container(border=True):
            st.markdown("**1 · Show**")
            c1, c2 = st.columns(2)
            show = c1.text_input("Show name", placeholder="e.g. My Show")
            genre = c2.text_input("Genre", placeholder="e.g. romance drama")
        with st.container(border=True):
            st.markdown("**2 · Screenplays**")
            files = st.file_uploader("Upload one file per episode", type=data.upload_types(pcfg.file_glob),
                                     accept_multiple_files=True)
            c1, c2 = st.columns([2, 1])
            folder = c1.text_input("...or a folder on this machine", placeholder="input/My Show")
            episodes = c2.text_input("Episodes (optional)", placeholder="all, or e.g. 1-5,8",
                                     help="Only these episodes: cheaper trial runs.")
        with st.container(border=True):
            st.markdown("**3 · Options**")
            c1, c2, c3, c4 = st.columns(4)
            review = c1.toggle("Editor review", value=settings.switches.editor_review,
                               help="Pause after ranking so you can approve, reject or fix each promo.")
            memory = c2.toggle("Memory", value=settings.switches.memory,
                               help="Use the rules learned from your earlier reviews.")
            joiner = c3.toggle("Joiner (multi-clip)", value=settings.switches.joiner,
                               help="Also build promos that join clips from different episodes.")
            dry = c4.toggle("Dry run", value=False, help="Fake agents: no API calls and no cost. For trying the app.")
        with st.expander("4 · Models (optional)", icon=":material/tune:"):
            st.caption("Main model per agent. Its backups from settings.yaml take over if its free quota runs out.")
            picked = {}
            cols = st.columns(len(AGENT_LABELS))
            for col, (agent, label) in zip(cols, AGENT_LABELS.items()):
                cur = settings.model_for(agent)
                cur_id = f"{cur.provider}:{cur.model}"
                picked[agent] = (col.selectbox(label, choices, index=choices.index(cur_id), key=f"model_{agent}"), cur_id)
        start = st.form_submit_button("Start run", type="primary", icon=":material/play_arrow:")
    if not start:
        return
    problems = [m for m, bad in [("Enter a show name.", not show.strip()), ("Enter a genre.", not genre.strip()),
                                 ("Upload screenplays or give a folder.", not files and not folder.strip())] if bad]
    if folder.strip() and not files and not (ROOT / folder.strip()).is_dir() and not Path(folder.strip()).is_dir():
        problems.append(f"Folder not found: {folder}")
    if episodes.strip():
        try:
            from run import parse_episodes
            parse_episodes(episodes)
        except ValueError:
            problems.append("Episodes must look like 1-5,8")
    if problems:
        for p in problems:
            st.error(p)
        return
    run_id = new_run_id(show)
    if files:
        input_dir = data.save_uploads(files, settings.resolve(settings.paths.uploads_dir) / run_id)
    else:
        f = Path(folder.strip())
        input_dir = f if f.is_absolute() else ROOT / f
    models = {a: v for a, (v, cur) in picked.items() if v != cur}
    data.launch(SETTINGS_PATH, run_id, input_dir, show.strip(), genre.strip(), agents="stub" if dry else "llm",
                review=review, memory=memory, joiner=joiner, episodes=episodes.strip() or None, models=models)
    st.session_state["goto"] = run_id
    time.sleep(1.0)                                  # let run.py create its folder
    st.rerun()


# ------------------------------------------------------------------------------ helpers
def promo_table(promos: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Rank": p["rank"], "Promo": p["promo_id"], "Type": p["moment_type"], "Segments": segs_text(p["segments"]),
        "Sec": p["total_duration_seconds"], "Final": p["scores"].get("final"), "Hook score": p["scores"].get("hook"),
        "Cliffhanger score": p["scores"].get("cliffhanger"), "Standalone score": p["scores"].get("standalone"),
        "Hook": p["hook"], "Cliffhanger": p["cliffhanger"], "Why it works": p["why_it_works"]} for p in promos])


SCORE_COLUMNS = {
    "Final": st.column_config.ProgressColumn("Final", min_value=0, max_value=10, format="%.2f"),
    **{c: st.column_config.ProgressColumn(c.replace(" score", ""), min_value=0, max_value=10, format="%.0f")
       for c in ("Hook score", "Cliffhanger score", "Standalone score")},
    "Hook": st.column_config.TextColumn("Hook", width="large"),
    "Cliffhanger": st.column_config.TextColumn("Cliffhanger", width="large"),
}


def render_trace(tr: dict | None, cand: dict | None = None):
    if not tr:
        st.info("No trace file yet for this promo (written when the run pauses, stops or finishes).")
        return
    final = tr.get("final") or {}
    st.subheader(f":material/timeline: {tr['promo_id']}")
    ui.chips([(final.get("moment_type") or "other", "violet"), (f"trace {tr['trace_id']}", "gray")])
    if cand:
        r = cand.get("ranking", {})
        ui.stat_cards([("Status", cand.get("status") or "-"), ("Final version", f"v{cand.get('final_version')}"),
                       ("Ranking", r.get("outcome", "-") + (f" #{r['rank']}" if r.get("rank") else "")),
                       ("Final score", (cand.get("final_scores") or {}).get("final", "-"))])
        if r.get("outcome") == "dropped":
            extra = {k: v for k, v in r.items() if k not in ("outcome", "reason", "final")}
            st.warning(f"Dropped at ranking: **{r.get('reason')}**" + (f" ({json.dumps(extra)})" if extra else ""),
                       icon=":material/block:")
    if tr.get("moments"):
        st.markdown("##### Reader moments used")
        st.dataframe(pd.DataFrame(tr["moments"])[["moment_id", "episode", "kind", "timestamp", "strength",
                                                    "moment_type", "description"]], hide_index=True)
    evals = {}
    for e in tr.get("evaluations") or []:
        evals.setdefault(e["version"], []).append(e)
    st.markdown("##### Versions and scores")
    for v in tr.get("versions") or []:
        ev = (evals.get(v["version"]) or [None])[-1]
        tag = f"final {ev['final']} · {'passed' if ev['passed'] else 'failed'}" if ev else "not evaluated"
        with st.expander(f"v{v['version']} · {segs_text(v['segments'])} · {v['total_duration_seconds']}s · {tag}",
                         expanded=v["version"] == final.get("version")):
            st.markdown(f"**Hook:** {v.get('hook')}  \n**Cliffhanger:** {v.get('cliffhanger')}  \n"
                        f"**Why it works:** {v.get('why_it_works')}")
            if v.get("join_rationale"):
                st.markdown(f"**Join:** {v['join_rationale']}")
            if v.get("revision_note"):
                st.markdown(f"**Revision ({v.get('revised_by', '')}):** {v['revision_note']}")
            if ev:
                s, rs = ev["scores"], ev.get("reasons") or {}
                st.markdown(f"**Scores:** hook {s['hook']} · cliffhanger {s['cliffhanger']} · standalone "
                            f"{s['standalone']} → **{ev['final']}**")
                for k in ("hook", "cliffhanger", "standalone"):
                    if rs.get(k):
                        st.caption(f"{k}: {rs[k]}")
                if ev.get("fix_instruction"):
                    st.warning(f"Fix asked: {ev['fix_instruction']}")
                if ev.get("llm"):
                    st.caption(f"Evaluator: {ev['llm'].get('provider')}:{ev['llm'].get('model')} · "
                               f"{ev['llm'].get('mode')} · retries {ev['llm'].get('retries')}")
                if ev.get("promo_text"):
                    st.code(ev["promo_text"], language=None)
    if tr.get("editor_decisions"):
        st.markdown("##### Editor decisions")
        st.dataframe(pd.DataFrame(tr["editor_decisions"]), hide_index=True)
    st.markdown("##### Event timeline")
    st.dataframe(pd.DataFrame([{
        "time": time.strftime("%H:%M:%S", time.localtime(e["ts"])), "node": e["node"], "event": e["event"],
        "details": json.dumps({k: v for k, v in (e.get("data") or {}).items() if k not in ("segments",)},
                              ensure_ascii=False, default=str)[:300]} for e in tr.get("events") or []]),
        hide_index=True)


# ---------------------------------------------------------------------------- run pages
def overview(run_id: str, run: dict):
    @st.fragment(run_every=5 if data.live(settings, run_id)["running"] else None)
    def live_panel():
        lv = data.live(settings, run_id)
        pr = lv.get("progress")
        if lv.get("checkpoint_state") == "awaiting_review" and not lv["running"]:
            st.info("This run is waiting for you: open the **Review** tab to approve, reject or fix its promos.",
                    icon=":material/rate_review:")
        if pr:
            stage = STAGES.get(pr["stage"], pr["stage"])
            st.progress(pr["fraction"], text=f"**{stage}** · {pr['fraction']:.0%}"
                        + (" · refreshing every 5 s" if lv["running"] else ""))
            ui.stat_cards([("Episodes read", f"{pr['read']}/{pr['total']}"),
                           ("Episodes cut", f"{pr['cut']}/{pr['total']}"),
                           ("Promos made", pr["candidates"], f"{pr['evaluated']} scored so far"),
                           ("Ranked", pr["ranked"]), ("Errors", pr["errors"])])
        if lv["running"]:
            if st.button("Stop run", icon=":material/stop_circle:",
                         help="Stops the process; finished steps are kept and Resume continues from there"):
                data.stop(settings, run_id)
                time.sleep(1)
                st.rerun()
        elif lv.get("checkpoint_state") == "stopped":
            if st.button("Resume run", type="primary", icon=":material/play_arrow:"):
                try:
                    data.resume(SETTINGS_PATH, run_id)
                except RunLocked as e:
                    st.error(str(e))
                else:
                    time.sleep(1)
                    st.rerun()
        tail = data.log_tail(settings, run_id)
        if tail:
            with st.expander("Log (latest lines)", icon=":material/terminal:", expanded=lv["running"]):
                st.code(tail, language=None)
    live_panel()

    s = run["summary"]
    if not s:
        return
    if s.get("stop_reason"):
        st.warning(f"Last stop: {s['stop_reason'][:500]}", icon=":material/pause_circle:")
    st.markdown("##### Settings")
    sw = s.get("switches") or {}
    ui.chips([ui.on_off(k.replace("_", " ").capitalize(), v) for k, v in sw.items()]
             + [(f"agents: {s.get('agents_kind')}", "violet")]
             + [(f"{a}: {m}", "violet") for a, m in (s.get("model_overrides") or {}).items()])
    u = s.get("llm_usage") or {}
    if u.get("by_agent"):
        t = u["totals"]
        st.markdown("##### AI usage")
        ui.stat_cards([("Calls", t["calls"]), ("Retry rate", u["retry_rate"]),
                       ("Tokens in", f"{t['input_tokens']:,}"), ("Tokens out", f"{t['output_tokens']:,}"),
                       ("Model switches", t["model_switches"], "a backup took over after a quota ran out")])
        with st.expander("Per agent", icon=":material/smart_toy:"):
            st.dataframe(pd.DataFrame([{"agent": a, "models": ", ".join(f"{m} ({n})" for m, n in x["models"].items()),
                                        "calls": x["calls"], "retries": x["retries"], "native": x["native_calls"],
                                        "json mode": x["fallback_calls"], "tokens in": x["input_tokens"],
                                        "tokens out": x["output_tokens"]} for a, x in u["by_agent"].items()]),
                         hide_index=True)
    if s.get("errors"):
        with st.expander(f"Errors ({len(s['errors'])})", icon=":material/error:"):
            for e in s["errors"]:
                st.caption(e[:400])


MIME = {".csv": "text/csv", ".json": "application/json", ".md": "text/markdown"}


def promos_tab(run_id: str, run: dict):
    promos = run["promos"]
    bar = st.container(horizontal=True, vertical_alignment="center")
    with bar:
        view = st.segmented_control("View", ["Cards", "Table"], default="Cards", key="promo_view", required=True,
                                    label_visibility="collapsed")
        if run["files"]:
            with st.popover("Download", icon=":material/download:"):
                for name, path in run["files"].items():
                    st.download_button(f"Download {name}", path.read_bytes(), file_name=f"{run_id}_{name}",
                                       mime=MIME.get(Path(name).suffix, "text/plain"), key=f"dl_{name}",
                                       on_click="ignore", width="stretch", icon=":material/description:")
    if not promos:
        st.info("No ranked promos yet. They appear when the run pauses for review, stops or finishes.",
                icon=":material/hourglass_top:")
        return
    by_id = {c["promo_id"]: c for c in run["candidates"]}
    if view == "Table":
        st.caption(f"{len(promos)} ranked promos · select a row for its details and full trace.")
        ev = st.dataframe(promo_table(promos), hide_index=True, on_select="rerun", selection_mode="single-row",
                          key="promo_table", column_config=SCORE_COLUMNS)
        rows = ev.selection.rows if ev and hasattr(ev, "selection") else []
        if rows:
            p = promos[rows[0]]
            with st.container(border=True):
                render_trace(data.load_trace(settings, run_id, p["promo_id"]), by_id.get(p["promo_id"]))
        return
    st.caption(f"{len(promos)} ranked promos, best first · **Details** opens a promo's full history.")
    key = f"detail_{run_id}"
    for p in promos:
        if ui.promo_card(p, key=f"det_{p['promo_id']}"):
            st.session_state[key] = None if st.session_state.get(key) == p["promo_id"] else p["promo_id"]
        if st.session_state.get(key) == p["promo_id"]:
            with st.container(border=True):
                render_trace(data.load_trace(settings, run_id, p["promo_id"]), by_id.get(p["promo_id"]))


def trace_tab(run_id: str, run: dict):
    cands = run["candidates"]
    if not cands:
        st.info("No promos yet.")
        return
    order = sorted(cands, key=lambda c: (c["ranking"].get("rank") or 999, c["promo_id"]))
    label = lambda c: (f"#{c['ranking']['rank']} " if c["ranking"].get("rank") else f"[{c['ranking'].get('reason') or c['ranking']['outcome']}] ") \
        + f"{c['promo_id']} · {(c.get('final_scores') or {}).get('final', '-')}"
    pick = st.selectbox("Promo (ranked first, then dropped)", range(len(order)), format_func=lambda i: label(order[i]))
    render_trace(data.load_trace(settings, run_id, order[pick]["promo_id"]), order[pick])


LATER = "(decide later)"


def _send_decisions(run_id: str, run: dict, rnd: int, decisions: dict):
    path = review.write_decisions(run["dir"], rnd, decisions)
    try:
        data.resume(SETTINGS_PATH, run_id, decisions=path)
    except RunLocked as e:
        st.error(str(e))
        return
    time.sleep(1)
    st.rerun()


def _scores_line(s: dict) -> str:
    return (f"final **{s.get('final')}** · hook {s.get('hook')} · cliffhanger {s.get('cliffhanger')} · "
            f"standalone {s.get('standalone')}")


def _review_card(it: dict, rnd: int):
    p, sc = it["promo"], it["scores"]
    top = st.container(horizontal=True, vertical_alignment="center", gap="small")
    top.markdown(f"#### #{it['rank']} · {it['promo_id']}")
    with top:
        st.badge(f"Score {sc.get('final')}", icon=":material/star:", color=ui.score_color(sc.get("final")))
    ui.chips([(p.get("moment_type") or "other", "violet"), (f"{p.get('total_duration_seconds')}s", "gray"),
              (segs_text(p["segments"]), "gray")])
    st.caption(f"Hook {sc.get('hook')} · Cliffhanger {sc.get('cliffhanger')} · Standalone {sc.get('standalone')}")
    ba = it.get("before_after")
    if ba:
        st.markdown(f"**Your fix:** {ba.get('editor_comment')}  \n"
                    f"**FeedbackAgent read it as:** {ba.get('fix_interpretation') or '_(not available: your words were sent as they were)_'}  \n"
                    f"**Maker's change:** {(ba.get('after') or {}).get('revision_note')}")
        left, right = st.columns(2)
        for col, side, text in ((left, "before", ba.get("before_text")), (right, "after", ba.get("after_text"))):
            v, ev = ba.get(side) or {}, ba.get(f"{side}_eval") or {}
            col.markdown(f"**{side.upper()}** · v{v.get('version')} · {segs_text(v.get('segments') or [])} · "
                         f"{v.get('total_duration_seconds')}s  \n{_scores_line({**ev.get('scores', {}), 'final': ev.get('final')})}")
            col.code(text or "(no transcript)", language=None)
    else:
        ui.quote("Hook", p.get("hook"))
        ui.quote("Cliffhanger", p.get("cliffhanger"))
        if p.get("why_it_works"):
            st.caption(f":material/lightbulb: {p['why_it_works']}")
        for k, v in (it.get("reasons") or {}).items():
            st.caption(f"Evaluator on {k}: {v}")
        with st.expander("What the viewer sees", icon=":material/visibility:"):
            st.code(it.get("transcript") or "(no transcript)", language=None)
    if it.get("note"):
        st.warning(it["note"], icon=":material/info:")
    if "fix" not in it["options"] and it["fix_attempts"]:
        st.caption(f"No fix attempts left ({it['fix_attempts']}/{it['max_fix_attempts']}).")
    st.radio("Your decision", [LATER] + it["options"], key=f"dec_{rnd}_{it['promo_id']}", horizontal=True,
             format_func=lambda o: review.OPTIONS.get(o, o))
    st.text_input("Comment (required for Fix; for Reject, why helps the memory learn)", key=f"com_{rnd}_{it['promo_id']}")


def review_tab(run_id: str, run: dict):
    lv = data.live(settings, run_id)
    if lv["running"]:
        st.info("The run is working (or applying your decisions). This tab fills in when it pauses for review.",
                icon=":material/progress_activity:")
        return
    state = lv.get("checkpoint_state")
    if state == "awaiting_review" and lv.get("interrupt"):
        payload = lv["interrupt"]
        rnd = payload["round"]
        items = review.review_items(payload, lv["values"])
        todo = [i for i in items if i["pending"]]
        st.subheader(f":material/rate_review: Round {rnd} of max {payload.get('max_rounds')} · {len(todo)} to review")
        st.caption("**Approve** keeps a promo. **Reject** drops it, and the next-best promo takes its place next "
                   "round. **Fix** sends your note to the FeedbackAgent and the agent that made the promo; it is "
                   "re-cut, re-scored and shown before/after next round.")
        if st.button("Approve all and continue", icon=":material/done_all:"):
            _send_decisions(run_id, run, rnd, {i["promo_id"]: {"decision": "approve", "comment": ""} for i in todo})
        with st.form(f"review_round_{rnd}"):
            for it in todo:
                with st.container(border=True):
                    _review_card(it, rnd)
            submitted = st.form_submit_button("Submit decisions", type="primary", icon=":material/send:")
        if submitted:
            decisions = {}
            for it in todo:
                choice = st.session_state.get(f"dec_{rnd}_{it['promo_id']}", LATER)
                if choice != LATER:
                    decisions[it["promo_id"]] = {"decision": choice,
                                                 "comment": (st.session_state.get(f"com_{rnd}_{it['promo_id']}") or "").strip()}
            errors = review.validate(items, decisions)
            if not decisions:
                errors.append("Choose a decision for at least one promo.")
            for e in errors:
                st.error(e)
            if not errors:
                _send_decisions(run_id, run, rnd, decisions)
        decided = [i for i in items if not i["pending"]]
        if decided:
            st.markdown(f"**Already decided ({len(decided)})**")
            st.dataframe(pd.DataFrame([{"Rank": i["rank"], "Promo": i["promo_id"],
                                        "Decision": (i["decision"] or {}).get("decision"),
                                        "Comment": (i["decision"] or {}).get("comment")} for i in decided]),
                         hide_index=True)
    elif state == "done":
        v = lv["values"]
        rs = v.get("review_summary") or {}
        if not v.get("lessons"):
            st.info("This run had no editor review (review switched off, or not reviewed).")
            return
        st.subheader(f":material/check_circle: Review finished after {rs.get('rounds', v.get('review_round'))} round(s)")
        if rs.get("unreviewed"):
            st.warning(f"Stopped at the max number of rounds; not reviewed in their final form: "
                       f"{', '.join(rs['unreviewed'])}")
        st.dataframe(pd.DataFrame([{"Round": l.get("round"), "Promo": l["promo_id"], "Decision": l["decision"],
                                    "Comment": l.get("comment"), "FeedbackAgent reading": l.get("fix_interpretation"),
                                    "Version": l.get("version"), "Final score": l.get("final")}
                                   for l in v["lessons"]]), hide_index=True)
    else:
        st.info("Nothing to review yet. Runs with editor review on pause here after ranking.",
                icon=":material/rate_review:")


def run_page(run_id: str):
    run = data.load_run(settings, run_id)
    s = run["summary"] or {}
    state = next((r["state"] for r in data.list_runs(settings) if r["run_id"] == run_id), None)
    st.header(f":material/movie: {s.get('show') or run_id}", divider="blue")
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        ui.state_badge(state)
        if s.get("genre"):
            st.badge(s["genre"], color="violet")
        st.caption(f"Run `{run_id}`" + (f" · last written {s['written_at']}" if s.get("written_at") else ""))
    t1, t2, t3, t4 = st.tabs([":material/dashboard: Overview", f":material/movie: Promos ({len(run['promos'])})",
                              ":material/timeline: Trace", ":material/rate_review: Review"])
    with t1:
        overview(run_id, run)
    with t2:
        promos_tab(run_id, run)
    with t3:
        trace_tab(run_id, run)
    with t4:
        review_tab(run_id, run)


# ------------------------------------------------------------------------------ memory
STATUS_LABEL = {"needs_review": "Waiting for you", "active": "Active", "retired": "Retired"}


def _rule_card(mem, r: dict):
    rid = r["id"]
    scope = r["scope"] + (f": {r['scope_value']}" if r.get("scope_value") else "")
    ui.chips([(f"R{rid}", "blue"), (scope, "violet"),
              (f"confidence {r['confidence']}", ui.score_color(10 * float(r["confidence"] or 0))),
              (f"+{r['supporting_count']} / -{r['contradicting_count']}", "gray"), (f"by {r['created_by']}", "gray")])
    st.caption(f"Updated {r['updated_at']}")
    if r.get("conflicts_with"):
        st.warning(f"Conflicts with {', '.join(f'R{x}' for x in r['conflicts_with'])}: decide which to keep.",
                   icon=":material/warning:")
    text = st.text_area("Rule", r["text"], key=f"text_{rid}", label_visibility="collapsed", height=70)
    if r["source_feedback_ids"]:
        with st.expander(f"Feedback behind it ({len(r['source_feedback_ids'])})"):
            for f in mem.feedback(r["source_feedback_ids"]):
                st.caption(f"#{f['id']} {f['show']} · {f['promo_id']} · {f['decision']}: \"{f['comment']}\"")
    c = st.columns([1, 1, 1, 1, 2])
    if r["status"] != "active" and c[0].button("Activate", key=f"activate_{rid}", type="primary", icon=":material/check:"):
        mem.set_status(rid, "active")
        st.rerun()
    if r["status"] != "retired" and c[1].button("Retire", key=f"retire_{rid}", icon=":material/archive:"):
        mem.set_status(rid, "retired")
        st.rerun()
    if c[2].button("Save text", key=f"save_{rid}", icon=":material/save:", disabled=text.strip() == r["text"]):
        mem.edit_rule(rid, text)
        st.rerun()
    confirm = c[4].checkbox("confirm delete", key=f"confirm_{rid}")
    if c[3].button("Delete", key=f"delete_{rid}", icon=":material/delete:"):
        if not confirm:
            st.warning("Tick 'confirm delete' first: deleting can't be undone (retiring can).")
        else:
            mem.delete_rule(rid)
            st.rerun()


def memory_page():
    from evaluate.runs import collect, memory_comparison
    from memory.store import MemoryStore
    mem = MemoryStore(settings.resolve(settings.memory.db_path))
    stats = mem.stats()
    ui.page_header("Editor memory", ":material/psychology:",
                   f"Guidelines {stats['version']} · learned from your reviews. Rules the AI suggests wait here until "
                   f"you activate them{' (auto-activate is ON in settings)' if settings.memory.auto_activate_rules else ''}; "
                   f"only active rules are added to the Reader, Cutter, Joiner and Evaluator prompts.")
    ui.stat_cards([("Waiting for you", stats["rules_needs_review"]), ("Active rules", stats["rules_active"]),
                   ("Retired", stats["rules_retired"]), ("Feedback items", stats["feedback"]),
                   ("Examples", stats["examples"])])
    by_status = {k: mem.rules(k) for k in STATUS_LABEL}
    tabs = st.tabs([f"{STATUS_LABEL[k]} ({len(v)})" for k, v in by_status.items()] +
                   ["Add rule", "Examples", "Feedback log", "Versions", "Runs & memory used"])
    for tab, (status, rules) in zip(tabs, by_status.items()):
        with tab:
            if not rules:
                st.info({"needs_review": "Nothing waiting. New rules appear here after reviewed runs.",
                         "active": "No active rules yet.", "retired": "No retired rules."}[status])
            for r in rules:
                with st.container(border=True):
                    _rule_card(mem, r)
    with tabs[3]:
        st.text_input("New rule (written by you, active at once)", key="new_rule_text")
        a, b = st.columns(2)
        scope = a.selectbox("Scope", ["global", "genre", "show"], key="new_rule_scope")
        value = b.text_input("Genre or show name (for genre/show scope)", key="new_rule_value")
        if st.button("Add rule", type="primary", icon=":material/add:"):
            text = (st.session_state.get("new_rule_text") or "").strip()
            if not text or (scope != "global" and not value.strip()):
                st.error("Write the rule, and give the genre or show for a genre/show rule.")
            else:
                mem.add_rule(text, scope=scope, scope_value=value.strip() or None, source_feedback_ids=[],
                             status="active", created_by="editor")
                st.rerun()
    with tabs[4]:
        ex = mem.examples()
        if ex:
            st.dataframe(pd.DataFrame([{"id": e["id"], "label": e["label"], "show": e["show"], "genre": e["genre"],
                                        "promo": e["promo"].get("promo_id"), "type": e["moment_type"],
                                        "final": e["final"], "reason": e["reason"], "hook": e["promo"].get("hook")}
                                       for e in ex]), hide_index=True)
        else:
            st.info("No examples yet.")
    with tabs[5]:
        fb = mem.feedback()
        st.caption("Append-only: every editor decision ever made. Never edited or deleted.")
        if fb:
            st.dataframe(pd.DataFrame([{k: f[k] for k in ("id", "timestamp", "show", "genre", "promo_id", "moment_type",
                                                            "decision", "comment", "guidelines_version", "run_id", "round")}
                                       for f in reversed(fb)]), hide_index=True)
        else:
            st.info("No feedback yet.")
    with tabs[6]:
        hist = mem.history()
        if not hist:
            st.info("No versions yet (v0 = empty memory).")
        for h in hist:
            with st.expander(f"v{h['version']} · {h['created_at']} · {h['reason']} · {len(h['active_rules'])} active"):
                for r in h["active_rules"]:
                    st.caption(f"R{r['id']} [{r['scope']}{':' + r['scope_value'] if r.get('scope_value') else ''}] {r['text']}")
    with tabs[7]:
        rows = []
        for r in data.list_runs(settings):
            s_ = (data.load_run(settings, r["run_id"])["summary"] or {})
            mu, mw = s_.get("memory_used") or {}, s_.get("memory_written") or {}
            rows.append({"run": r["run_id"], "memory": "on" if mu.get("enabled") else "off",
                         "guidelines used": mu.get("guidelines_version"), "rules injected": mu.get("rule_ids"),
                         "examples injected": mu.get("example_ids"),
                         "feedback written": mw.get("feedback_logged"), "new rules": mw.get("new_rule_ids"),
                         "guidelines after": mw.get("version_after")})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        cmp_ = memory_comparison(collect(settings))
        st.markdown("**Editor approval (first review round) with vs without memory**")
        st.dataframe(pd.DataFrame([{"": k.replace("_", " "), **v} for k, v in cmp_.items() if isinstance(v, dict)]),
                     hide_index=True)
        if cmp_["difference"] is not None:
            st.metric("Difference (with - without)", f"{cmp_['difference']:+.0%}")
        else:
            st.caption("Needs reviewed runs both with and without memory (toggle Memory when starting a run).")


# -------------------------------------------------------------------------- evaluation
VERDICT_STYLE = {"trust": st.success, "dont_rely": st.error}       # too_few / no_overlap -> st.info


def _pct(x) -> str:
    return "-" if x is None else f"{x:.0%}"


def _download(path: Path, key: str):
    if path.exists():
        st.download_button("Download report (.md)", path.read_text(encoding="utf-8"), file_name=path.name,
                           mime="text/markdown", key=key)


def _howto():
    lab_dir = Path(settings.evaluation.dir) / "labels"
    st.markdown(
        "Reports are made on the command line (labelling has to stay blind, so it never happens next to the "
        "tool's picks). This page only shows them.\n\n"
        f"1. **Label episodes yourself:** `python -m evaluate.labels --input <folder> --show <show> --episodes 2-5`\n"
        f"2. **AI labeller** (`{settings.models['labeler'].provider}:{settings.models['labeler'].model}`): "
        f"`python -m evaluate.ai_labels --input <folder> --show <show> --genre <genre> --episodes 2-9`\n"
        f"3. **Tool vs labels:** `python -m evaluate.compare --run <run_id>`\n"
        f"4. **Can the labeller be trusted?** `python -m evaluate.agreement --reference {lab_dir}/<show>.yaml "
        f"--candidate {lab_dir}/<show>-ai.yaml`\n"
        f"5. **A show with known answers:** `python -m evaluate.synthetic_show --show <name> --genre <genre> "
        f"--episodes 8`, then run the tool on `input/synthetic/<name>` and compare.")


def run_eval_tab(evals: list[dict]):
    from evaluate.compare import source_notes
    if not evals:
        st.info("No run evaluated yet: `python -m evaluate.compare --run <run_id>` (needs a labels file for the show).")
        return
    i = st.selectbox("Run", range(len(evals)), format_func=lambda k: evals[k]["run_id"], key="eval_run")
    r, rc = evals[i]["report"], evals[i]["report"]["recall"]
    st.caption(f"Labels: {r['labeled_by']} · locked {r.get('labels_locked_at') or 'NOT LOCKED'} · "
               f"a match = shared screen time >= {r['match_min_overlap']:.0%} of the shorter clip")
    for level, note in source_notes(r):
        (st.warning if level == "warning" else st.info)(note)
    st.markdown(f"**Recall funnel:** {rc['labels']} labelled moments in {rc['episodes_labelled']} episodes")
    c = st.columns(3)
    for col, (label, k) in zip(c, [("Reader marked it", "reader"), ("A promo was made", "candidates"),
                                   ("Made the final list", "ranked")]):
        col.metric(label, _pct(rc[f"{k}_rate"]), help=f"{rc[k]} of {rc['labels']}", border=True)
    st.dataframe(pd.DataFrame([{"label": k, "ep": x["episode"], "span": x["span"], "type": x["moment_type"],
                                "reader moments": ", ".join(x["reader_moments"]) or "-", "made by": x["made_by"] or "-",
                                "ranked by": x["ranked_by"] or "-", "lost at": x["stage_lost"] or "found",
                                "note": x["note"]} for k, x in r["per_label"].items()]), hide_index=True)
    pr, pk = r["post_rate"], r["picks_in_labelled_episodes"]
    st.caption(f"Would post: {pr['would_post']} of {pr['judged']} judged (post rate {_pct(pr['rate'])}) · "
               f"ranked picks in labelled episodes: {pk['picks']}, of which {pk['match_a_label']} match a label "
               f"({_pct(pk['rate'])})")
    _download(evals[i]["md"], key=f"dl_eval_{evals[i]['run_id']}")


def agreement_tab(reports: list[dict]):
    if not reports:
        st.info("No agreement check yet: `python -m evaluate.agreement --reference <trusted labels> "
                "--candidate <AI labels>`.")
        return
    i = st.selectbox("Check", range(len(reports)), format_func=lambda k: reports[k]["name"].replace("__vs__", "  vs  "),
                     key="eval_agreement")
    r = reports[i]["report"]
    st.caption(f"Reference (trusted): {r['reference']} [{r['reference_source']}]")
    st.caption(f"Candidate (checked): {r['candidate']} [{r['candidate_source']}] · episodes in both: "
               f"{data.episode_ranges(r['episodes']) or 'none'} · written {r.get('written_at', '')}")
    for w in r["warnings"]:
        st.warning(w)
    VERDICT_STYLE.get(r["verdict"]["key"], st.info)(f"**{r['verdict']['text']}**")
    c = st.columns(4)
    c[0].metric("Recall", _pct(r["recall"]), border=True, help=f"{r['matched']} of {r['reference_moments']} reference moments found")
    c[1].metric("Precision", _pct(r["precision"]), border=True, help=f"{r['matched']} of {r['candidate_moments']} candidate moments are in the reference")
    c[2].metric("F1", "-" if r["f1"] is None else f"{r['f1']:.2f}", border=True)
    c[3].metric("Reference moments", r["reference_moments"], border=True)
    st.dataframe(pd.DataFrame([{"episode": e, **x} for e, x in r["per_episode"].items()]), hide_index=True)
    for title, key in (("Matched", "pairs"), ("Missed by the candidate", "missed"), ("Extra (candidate only)", "extra")):
        with st.expander(f"{title} ({len(r[key])})"):
            if r[key]:
                st.dataframe(pd.DataFrame(r[key]), hide_index=True)
            else:
                st.caption("None.")
    _download(reports[i]["md"], key=f"dl_agree_{reports[i]['name']}")


def label_files_tab(files: list[dict]):
    if not files:
        st.info("No labels yet: `python -m evaluate.labels ...` (you) or `python -m evaluate.ai_labels ...` (AI).")
        return
    st.caption("human = an editor labelled it blind · ai = the AI labeller · planted = a made-up show's answer key, "
               "exact by construction. Only fully labelled episodes count.")
    st.dataframe(pd.DataFrame([{k: v for k, v in f.items() if k != "path"} for f in files]), hide_index=True)


def evaluation_page():
    ui.page_header("Evaluation", ":material/insights:",
                   "How good are the picks, and how far can the labels behind that number be trusted?")
    evals, reports, files = (data.list_run_evaluations(settings), data.list_agreements(settings),
                             data.list_label_files(settings))
    t1, t2, t3, t4 = st.tabs([":material/fact_check: Tool vs labels", ":material/handshake: Labeller agreement",
                              ":material/label: Label files", ":material/help: How to add more"])
    with t1:
        run_eval_tab(evals)
    with t2:
        agreement_tab(reports)
    with t3:
        label_files_tab(files)
    with t4:
        _howto()


page, choice = sidebar()
if page == "evaluation":
    evaluation_page()
elif page == "memory":
    memory_page()
elif page == "runs" and choice:
    run_page(choice)
else:
    new_run_page()
