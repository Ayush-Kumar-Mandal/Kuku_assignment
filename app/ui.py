"""Small display helpers shared by the app's pages: status badges, setting chips, stat
cards, score badges and the promo card. Colours come from the theme in .streamlit/config.toml
(Streamlit's named colours: blue = the indigo accent, green, orange, red, violet, gray)."""
from __future__ import annotations

import streamlit as st

from graph.export import segs_text

# run state -> (label, badge colour, icon)
STATES = {
    "running": ("Running", "blue", ":material/progress_activity:"),
    "starting": ("Starting", "blue", ":material/hourglass_top:"),
    "awaiting_review": ("Waiting for your review", "orange", ":material/rate_review:"),
    "done": ("Done", "green", ":material/check_circle:"),
    "stopped": ("Stopped", "gray", ":material/pause_circle:"),
    "not_started": ("Not started", "gray", ":material/radio_button_unchecked:"),
    "unknown": ("Unknown", "gray", ":material/help:"),
}
STATE_DOT = {"blue": "🔵", "orange": "🟠", "green": "🟢", "gray": "⚪"}


def state_info(state: str | None) -> tuple[str, str, str]:
    return STATES.get(state or "unknown", (str(state), "gray", ":material/help:"))


def state_badge(state: str | None):
    label, color, icon = state_info(state)
    st.badge(label, icon=icon, color=color)


def state_dot(state: str | None) -> str:
    """One-character status for places that can't hold a badge (select boxes)."""
    return STATE_DOT[state_info(state)[1]]


def chips(items: list[tuple[str, str]]):
    """A row of small badges: [(text, colour), ...]."""
    with st.container(horizontal=True, gap="small"):
        for text, color in items:
            st.badge(text, color=color)


def on_off(name: str, value) -> tuple[str, str]:
    return (f"{name} {'on' if value else 'off'}", "blue" if value else "gray")


def stat_cards(items: list[tuple[str, object]] | list[tuple[str, object, str | None]]):
    """Bordered metric cards in one row: (label, value[, help])."""
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        label, value, help_ = (*it, None)[:3]
        col.metric(label, value, help=help_, border=True)


def score_color(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "gray"
    return "green" if x >= 7.5 else "blue" if x >= 6.5 else "orange" if x >= 5 else "red"


def page_header(title: str, icon: str, subtitle: str = ""):
    st.header(f"{icon} {title}", divider="blue")
    if subtitle:
        st.caption(subtitle)


def quote(label: str, text: str | None):
    if text:
        st.markdown(f"**{label}**  \n> {text}")


def promo_card(p: dict, key: str) -> bool:
    """One ranked promo as a card. Returns True when its Details button is clicked."""
    s = p.get("scores") or {}
    with st.container(border=True):
        top = st.container(horizontal=True, vertical_alignment="center", gap="small")
        top.markdown(f"#### #{p.get('rank')} · {p['promo_id']}")
        with top:
            final = s.get("final")
            st.badge(f"Score {final:.2f}" if isinstance(final, (int, float)) else "Score -",
                     icon=":material/star:", color=score_color(final))
        chips([(p.get("moment_type") or "other", "violet"),
               (f"{p.get('total_duration_seconds')}s", "gray"),
               (segs_text(p.get("segments") or []), "gray")])
        quote("Hook", p.get("hook"))
        quote("Cliffhanger", p.get("cliffhanger"))
        if p.get("why_it_works"):
            st.caption(f":material/lightbulb: {p['why_it_works']}")
        bottom = st.container(horizontal=True, vertical_alignment="center")
        bottom.caption(f"Hook {s.get('hook', '-')} · Cliffhanger {s.get('cliffhanger', '-')} · "
                       f"Standalone {s.get('standalone', '-')}")
        return bottom.button("Details", key=key, icon=":material/expand_content:")
