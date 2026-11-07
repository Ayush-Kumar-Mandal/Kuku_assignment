"""Pydantic output models for every agent (what the LLM must return).

Kept provider-friendly: every field is required (use "" for "none") and there are no
nested optionals, so the same schema works with Gemini and with Groq strict json_schema.
Ids, versions, episode numbers of single-episode cuts and durations are NOT asked of the
model: code assigns or computes them from the screenplay, so they can't be wrong.
Field order puts reasoning before scores where it matters (models fill fields in order).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TS = "copied exactly from the screenplay's start or end column, MM:SS.ff"


class Segment(BaseModel):
    episode: int = Field(ge=1, description="Episode number")
    start_timestamp: str = Field(description=f"Start, {TS}")
    end_timestamp: str = Field(description=f"End, {TS}")


# ---------------------------------------------------------------------------- reader
class ReaderMoment(BaseModel):
    kind: Literal["hook", "cliffhanger"]
    timestamp: str = Field(description="START time of the line where the moment happens, copied exactly from the start column")
    quote: str = Field(description="first few words of that line, so a human can find it")
    description: str = Field(description="one sentence: what happens and why it grabs")
    moment_type: str = Field(description="one of the listed moment types")
    strength: int = Field(ge=1, le=10, description="how strong as a promo hook / cliffhanger")


class ReaderOutput(BaseModel):
    moments: list[ReaderMoment]
    new_viewer_context: str = Field(description="1-2 sentences a viewer starting at this episode needs to follow it")
    story_summary: str = Field(description="the running story summary, updated with this episode")


# ---------------------------------------------------------------------------- cutter
class CutPromo(BaseModel):
    start_timestamp: str = Field(description=f"clip start: a line's START time, {TS}")
    end_timestamp: str = Field(description=f"clip end: a line's END time, {TS}")
    hook_timestamp: str = Field(description="start time of the line that hooks the viewer (must be within the first seconds)")
    cliffhanger_timestamp: str = Field(description="start time of the line the clip leaves hanging on (near the end)")
    hook_moment_id: str = Field(description='id of the Reader moment used as the hook, or "" if none')
    cliffhanger_moment_id: str = Field(description='id of the Reader moment used as the cliffhanger, or "" if none')
    hook: str = Field(description="what the viewer sees/hears first, one short sentence")
    cliffhanger: str = Field(description="the unresolved beat the clip ends on, one short sentence")
    moment_type: str = Field(description="one of the listed moment types")
    why_it_works: str = Field(description="1-2 sentences for an editor")


class CutterOutput(BaseModel):
    promos: list[CutPromo]


class RevisedCut(CutPromo):
    revision_note: str = Field(description="what you changed and why, one sentence")


# ---------------------------------------------------------------------------- joiner
class JoinPromo(BaseModel):
    segments: list[Segment] = Field(description="2 or more clips, in the order they play")
    hook_timestamp: str = Field(description="start time of the hook line, inside the FIRST segment, near its start")
    cliffhanger_timestamp: str = Field(description="start time of the cliffhanger line, inside the LAST segment, near its end")
    hook_moment_id: str = Field(description='id of the Reader moment used as the hook, or ""')
    cliffhanger_moment_id: str = Field(description='id of the Reader moment used as the cliffhanger, or ""')
    hook: str
    cliffhanger: str
    moment_type: str
    why_it_works: str
    join_rationale: str = Field(description="why these clips belong together and play as one story")


class JoinerOutput(BaseModel):
    promos: list[JoinPromo]


class RevisedJoin(JoinPromo):
    revision_note: str = Field(description="what you changed and why, one sentence")


# ---------------------------------------------------------------------------- evaluator
class EvalOutput(BaseModel):
    hook_reason: str = Field(description="what happens in the first seconds and whether it stops the scroll")
    hook: int = Field(ge=1, le=10)
    cliffhanger_reason: str = Field(description="what question the ending leaves open")
    cliffhanger: int = Field(ge=1, le=10)
    standalone_reason: str = Field(description="can a first-time viewer follow who is who and what is at stake")
    standalone: int = Field(ge=1, le=10)
    passed: bool
    fix_instruction: str = Field(description='if not passed: one concrete edit, quoting the lines involved; "" if passed')


# ---------------------------------------------------------------------- feedback (Phase 2)
class FixPlan(BaseModel):
    interpretation: str = Field(description="one sentence: what the editor wants changed, in plain words")
    instruction: str = Field(description="the precise edit for the clip maker, citing real start/end times from the screenplay")
    keep: str = Field(description='what must stay as it is (e.g. the opening line), or ""')


# ---------------------------------------------------------------------- learning (Phase 3)
class FeedbackRelation(BaseModel):
    feedback_id: int = Field(description="id of a NEW feedback item (#id)")
    rule_id: int = Field(description="id of an existing rule (Rid)")
    relation: Literal["supports", "contradicts"]
    why: str = Field(description="a few words")


class ProposedRule(BaseModel):
    text: str = Field(description="one general, actionable rule for cutting promos, not about a single promo")
    source_feedback_ids: list[int] = Field(description="ids of the feedback items (#id) that back it")
    conflicts_with: list[int] = Field(description="ids of existing rules it contradicts, or []")
    why: str = Field(description="a few words")


class RuleConflict(BaseModel):
    rule_a: int
    rule_b: int
    why: str


class LearnOutput(BaseModel):
    relations: list[FeedbackRelation]
    new_rules: list[ProposedRule]
    conflicts: list[RuleConflict]


# ---------------------------------------------------------------------- AI labeller (evaluation only)
class LabelerMoment(BaseModel):
    start: str = Field(description="START time of the stretch's first line, copied exactly")
    end: str = Field(description="END time of the stretch's last line, copied exactly")
    moment_type: str
    note: str = Field(description="the opening line and the closing beat, in a few words")
    why: str = Field(description="one sentence: why it opens strong, ends open and stands alone")


class LabelerOutput(BaseModel):
    moments: list[LabelerMoment] = Field(description="every qualifying stretch, best first; [] if none")


# ---------------------------------------------------------------------- synthetic show writer (evaluation only)
class BibleCharacter(BaseModel):
    name: str = Field(description="first name or title, as it would appear in a script")
    role: str = Field(description="e.g. heroine, rival, mentor")
    description: str = Field(description="one sentence: who they are and what they want")


class ShowBible(BaseModel):
    premise: str = Field(description="2-3 sentences: the central conflict of the show")
    characters: list[BibleCharacter] = Field(min_length=2)
    episode_beats: list[str] = Field(description="one line per episode: what happens in it, in order")


class DraftLine(BaseModel):
    kind: Literal["action", "dialogue", "sound"]
    speaker: str = Field("", description="dialogue only: who speaks")
    emotion: str = Field("", description="dialogue only: 1-2 words, e.g. Cold, Trembling")
    text: str = Field(description="one action, one spoken line, or one sound cue")
    plant: str | None = Field(None, description="the planted moment id (e.g. P1) this line belongs to, else null")


class DraftScene(BaseModel):
    heading: str = Field(description="e.g. INT. MANOR HALL - NIGHT")
    lines: list[DraftLine] = Field(min_length=1)


class EpisodeDraft(BaseModel):
    # summary comes before the long scenes list: models tend to drop fields after it
    logline: str = Field(description="one sentence")
    continuity_notes: str = Field("", description="one sentence: what carries over from earlier episodes")
    summary: str = Field("", description="2-3 sentences: what happened, for writing the next episodes")
    scenes: list[DraftScene] = Field(min_length=1)


class ScriptScene(BaseModel):
    heading: str = Field(description="e.g. INT. MANOR HALL - NIGHT")
    lines: list[str] = Field(min_length=1, description='one string per line: "ACTION: ...", "SOUND: ...", or '
                                                       '"NAME (Emotion): spoken text"; planted lines start with "[P1] "')


class EpisodeScript(BaseModel):
    """What the synthetic writer answers: plain screenplay-style lines (compact, few JSON
    keys). Code turns it into an EpisodeDraft (evaluate.synthetic_show.to_draft)."""
    logline: str = Field(description="one sentence")
    continuity_notes: str = Field("", description="one sentence: what carries over from earlier episodes")
    summary: str = Field("", description="2-3 sentences: what happened, for writing the next episodes")
    scenes: list[ScriptScene] = Field(min_length=1)
