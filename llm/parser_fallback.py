"""LLM fallback for screenplay lines the regex patterns couldn't parse (spec §7, Parser).

Plugs into parse_episode(..., fallback=make_parser_fallback(client, parser_cfg)).
Every call goes through LLMClient, so it is logged and counted like any agent call.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from config.loader import ParserConfig
from llm.client import LLMClient, StructuredOutputError
from llm.prompts import load_chat_prompt
from parser.models import Episode, Item, UnparsedLine
from parser.screenplay_parser import ParseError, parse_ts


class RecoveredItem(BaseModel):
    line_no: int = Field(description="line number of the unparsed line this item comes from")
    kind: str = Field(description="item kind")
    character: str | None = Field(None, description="speaker, dialogue only")
    text: str
    start_timestamp: str = Field(description="MM:SS.ff")
    end_timestamp: str = Field(description="MM:SS.ff")


class RecoveredItems(BaseModel):
    items: list[RecoveredItem] = []


def _context(ep: Episode, line_no: int) -> str:
    items = ep.all_items()
    before = [it for it in items if it.line_no < line_no][-1:]
    after = [it for it in items if it.line_no > line_no][:1]
    return " | ".join(f"{'before' if it in before else 'after'}: {it.render()}" for it in before + after) or "none"


def make_parser_fallback(client: LLMClient, parser_cfg: ParserConfig, agent: str = "parser_fallback"):
    kinds = sorted(set(parser_cfg.block_kinds.values()) | {parser_cfg.dialogue_kind})
    prompt = load_chat_prompt(client.prompts_dir, "parser_fallback")
    pats = parser_cfg.compiled

    def fallback(unparsed: list[UnparsedLine], episode: Episode) -> list[Item]:
        valid_lines = {u.line_no for u in unparsed}

        def check(out: RecoveredItems) -> list[str]:
            probs = []
            for r in out.items:
                if r.line_no not in valid_lines:
                    probs.append(f"line_no {r.line_no} is not one of the unparsed lines {sorted(valid_lines)}")
                if r.kind not in kinds:
                    probs.append(f"line {r.line_no}: kind '{r.kind}' not in {kinds}")
                for ts in (r.start_timestamp, r.end_timestamp):
                    try:
                        parse_ts(ts, pats)
                    except ParseError:
                        probs.append(f"line {r.line_no}: '{ts}' is not MM:SS.ff")
            return probs

        lines = "\n".join(f"line {u.line_no}: {u.text!r}  (context {_context(episode, u.line_no)})" for u in unparsed)
        try:
            res = client.structured(agent, prompt, {"episode": episode.number, "lines": lines, "kinds": ", ".join(kinds)},
                                    RecoveredItems, checks=[check], label=f"parse ep{episode.number}")
        except StructuredOutputError:
            return []   # lines stay in `unparsed`; already logged by the client
        out = []
        for n, r in enumerate(res.output.items, start=1):
            s, e = parse_ts(r.start_timestamp, pats), parse_ts(r.end_timestamp, pats)
            out.append(Item(item_id=f"fallback_{n}", episode=episode.number, scene=0, index=10_000 + n,
                            kind=r.kind, label=r.character or r.kind.upper(), character=r.character,
                            text=r.text, start=s, end=max(e, s), start_ts=r.start_timestamp,
                            end_ts=r.end_timestamp, line_no=r.line_no))
        return out

    return fallback
