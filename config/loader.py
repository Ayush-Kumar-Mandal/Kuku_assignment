"""Load the YAML config files into typed Pydantic models.

Every model uses extra="forbid" so a typo'd or stale key fails at load time with a clear
message instead of being silently ignored mid-run.
"""
from __future__ import annotations

import re
from functools import cached_property
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS = ROOT / "config" / "settings.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- settings
class ModelSpec(_Strict):
    provider: str
    model: str
    temperature: float | None = Field(None, ge=0, le=2)   # None = provider default
    structured_output: Literal["auto", "native", "fallback"] = "auto"
    kwargs: dict = {}
    # Backup models, tried in order when this one's quota runs out (daily quota, or still
    # rate-limited/overloaded after every backoff wait). The switch lasts for the rest of the run.
    fallbacks: list["ModelSpec"] = []

    @model_validator(mode="after")
    def _flat(self):
        if any(f.fallbacks for f in self.fallbacks):
            raise ValueError("a fallback model cannot have fallbacks of its own; list them all in order")
        return self

    def chain(self) -> list["ModelSpec"]:
        return [self.model_copy(update={"fallbacks": []})] + list(self.fallbacks)


class ProviderSpec(_Strict):
    api_key_env: str
    native_method: str = "json_schema"
    json_mode_kwargs: dict = {}


class LLMSettings(_Strict):
    max_retries: int = Field(ge=0)
    request_timeout_s: int = Field(gt=0)
    provider_max_retries: int = Field(1, ge=0)
    rate_limit_backoff_s: list[float]
    max_rate_limit_waits: int = Field(30, ge=1)


class PromoSettings(_Strict):
    min_duration_s: float = Field(gt=0)
    max_duration_s: float = Field(gt=0)
    hook_window_s: float = Field(gt=0)
    max_segments: int = Field(ge=1)
    boundary_tolerance_s: float = Field(ge=0)
    cliffhanger_window_s: float = Field(gt=0)

    @model_validator(mode="after")
    def _range(self):
        if self.min_duration_s >= self.max_duration_s:
            raise ValueError("promo.min_duration_s must be < promo.max_duration_s")
        return self


class ReaderAgentSettings(_Strict):
    max_hooks: int = Field(ge=1)
    max_cliffhangers: int = Field(ge=1)
    summary_max_words: int = Field(ge=20)


class CutterAgentSettings(_Strict):
    max_promos_per_episode: int = Field(ge=1)


class JoinerAgentSettings(_Strict):
    max_promos: int = Field(ge=0)
    max_moments: int = Field(ge=2)
    min_moment_strength: int = Field(ge=1, le=10)
    context_before_s: float = Field(ge=0)
    context_after_s: float = Field(gt=0)


class ReviseAgentSettings(_Strict):
    context_s: float = Field(ge=0)


class AgentSettings(_Strict):
    screenplay_kinds: list[str] = Field(min_length=1)
    reader: ReaderAgentSettings
    cutter: CutterAgentSettings
    joiner: JoinerAgentSettings
    revise: ReviseAgentSettings


class EvaluatorSettings(_Strict):
    max_revision_rounds: int = Field(ge=0)
    pass_threshold: float = Field(ge=1, le=10)
    min_each_score: int = Field(ge=1, le=10)
    batch_size: int = Field(5, ge=1)


class ScoreWeights(_Strict):
    hook: float = Field(ge=0)
    cliffhanger: float = Field(ge=0)
    standalone: float = Field(ge=0)

    @model_validator(mode="after")
    def _sum(self):
        total = self.hook + self.cliffhanger + self.standalone
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"scoring.weights must sum to 1.0 (got {total})")
        return self


class ScoringSettings(_Strict):
    weights: ScoreWeights


class RankingSettings(_Strict):
    top_n: int = Field(ge=1)
    max_per_episode: int = Field(ge=1)
    overlap_threshold: float = Field(ge=0, le=1)
    early_episode_preference: float = Field(ge=0, le=1)
    min_final_score: float = Field(ge=0, le=10)


class FeedbackSettings(_Strict):
    max_fix_attempts: int = Field(ge=0)
    backfill_rejected: bool = True                 # re-rank after review: next promo fills a rejected slot
    max_review_rounds: int = Field(5, ge=1)        # pauses for the editor, at most


class MemorySettings(_Strict):
    db_path: str
    top_rules: int = Field(ge=0)
    approved_examples: int = Field(ge=0)
    rejected_examples: int = Field(ge=0)
    min_feedback_for_rule: int = Field(ge=1)
    auto_activate_rules: bool = False     # False: AI-proposed rules wait for the editor (needs_review)
    write_when_off: bool = True           # memory switch off = nothing injected; feedback still recorded
    learn_window: int = Field(200, ge=0)  # earlier unused feedback items the learner may build rules from


class Switches(_Strict):
    memory: bool
    joiner: bool
    editor_review: bool


class GraphSettings(_Strict):
    recursion_limit: int = Field(1000, ge=25)
    durability: Literal["sync", "async", "exit"] = "sync"


class AILabelerSettings(_Strict):
    max_per_episode: int = Field(4, ge=1)
    story_loglines: int = Field(12, ge=0)


class AgreementSettings(_Strict):
    trust_min_recall: float = Field(0.7, ge=0, le=1)
    trust_min_precision: float = Field(0.5, ge=0, le=1)
    min_reference_moments: int = Field(10, ge=1)


class EvaluationSettings(_Strict):
    dir: str
    match_min_overlap: float = Field(gt=0, le=1)
    failure_categories: dict[str, str] = Field(min_length=1)
    ai_labeler: AILabelerSettings = AILabelerSettings()
    agreement: AgreementSettings = AgreementSettings()


class ModelPrice(_Strict):
    input: float = Field(ge=0)       # USD per 1M input tokens
    output: float = Field(ge=0)      # USD per 1M output tokens (incl. thinking tokens)


class FreeLimit(_Strict):
    rpd: int | None = Field(None, ge=1)     # requests per day; None = unknown
    tpd: int | None = Field(None, ge=1)     # tokens per day; None = none / unknown
    source: str = ""


class CostSettings(_Strict):
    prices_checked: str
    shows_per_week: int = Field(ge=1)
    episodes_per_show: int | None = Field(None, ge=1)    # None = same as the measured run
    paid_prices_per_m: dict[str, ModelPrice]             # key: provider:model
    free_limits: dict[str, FreeLimit]


class Paths(_Strict):
    output_dir: str
    logs_dir: str
    checkpoint_db: str
    parser_config: str
    moment_types: str
    prompts_dir: str
    uploads_dir: str = "input/uploads"     # UI uploads land here (gitignored with input/)
    synthetic_config: str = "config/synthetic.yaml"   # evaluate.synthetic_show


class Settings(_Strict):
    models: dict[str, ModelSpec]
    providers: dict[str, ProviderSpec]
    llm: LLMSettings
    promo: PromoSettings
    agents: AgentSettings
    evaluator: EvaluatorSettings
    scoring: ScoringSettings
    ranking: RankingSettings
    feedback: FeedbackSettings
    memory: MemorySettings
    switches: Switches
    evaluation: EvaluationSettings
    cost: CostSettings
    graph: GraphSettings = GraphSettings()
    paths: Paths

    @field_validator("models")
    @classmethod
    def _has_default(cls, v):
        if "default" not in v:
            raise ValueError("models must define a 'default' entry")
        return v

    @model_validator(mode="after")
    def _providers_known(self):
        missing = {c.provider for m in self.models.values() for c in m.chain()} - set(self.providers)
        if missing:
            raise ValueError(f"models use providers with no 'providers' entry: {sorted(missing)}")
        return self

    def model_for(self, agent: str) -> ModelSpec:
        """Agent-specific model, falling back to models.default."""
        return self.models.get(agent, self.models["default"])

    def resolve(self, rel: str) -> Path:
        """Resolve a config-relative path against the project root."""
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p


# ----------------------------------------------------------------------- moment types
class MomentTypes(_Strict):
    moment_types: dict[str, str]
    default_type: str

    @model_validator(mode="after")
    def _default_listed(self):
        if self.default_type not in self.moment_types:
            raise ValueError("default_type must be one of moment_types")
        return self

    @property
    def names(self) -> list[str]:
        return list(self.moment_types)


# ----------------------------------------------------------------------------- parser
_FLAG_NAMES = Literal["IGNORECASE", "MULTILINE", "UNICODE", "VERBOSE"]


class HeaderPatterns(_Strict):
    episode_number: str
    logline: str
    continuity_notes: str


class ScenePatterns(_Strict):
    number: str
    heading: str
    start: str
    end: str


class TimestampSpec(_Strict):
    pattern: str


class SpeakerSpec(_Strict):
    name: str
    parenthetical: str
    emotion_separator: str = ","
    voice_tags: list[str] = []


class ParserConfig(_Strict):
    model_config = ConfigDict(extra="forbid", ignored_types=(cached_property,))

    file_glob: str = "*.txt"
    encoding: str = "utf-8"
    flags: list[_FLAG_NAMES] = []
    episode_number_from_filename: str
    timestamp: TimestampSpec
    header: HeaderPatterns
    script_start: str
    script_end: str
    scene: ScenePatterns
    ignore: list[str] = []
    block_header: str
    timing_line: str
    block_kinds: dict[str, str]
    dialogue_kind: str = "dialogue"
    speaker: SpeakerSpec
    dialogue_quotes: str = '"'
    warning_tolerance_s: float = 0.05
    block_ends_on_blank_line: bool = True

    @model_validator(mode="after")
    def _compile_all(self):
        # Compile every pattern once at load so a bad regex fails here, not mid-parse.
        _ = self.compiled
        return self

    @property
    def re_flags(self) -> int:
        f = 0
        for name in self.flags:
            f |= getattr(re, name)
        return f

    @cached_property
    def compiled(self) -> "CompiledPatterns":
        return CompiledPatterns(self)


class CompiledPatterns:
    """All parser regexes, compiled, with `{ts}` expanded."""

    def __init__(self, cfg: ParserConfig):
        flags = cfg.re_flags
        # The timestamp pattern is embedded in other patterns; its named groups would
        # clash there (two timestamps on one line), so embed it with plain groups.
        ts_inline = re.sub(r"\(\?P<\w+>", "(?:", cfg.timestamp.pattern)

        def c(p: str, name: str) -> re.Pattern:
            try:
                return re.compile(p.replace("{ts}", ts_inline), flags)
            except re.error as e:
                raise ValueError(f"parser.yaml: invalid regex for '{name}': {e}") from e

        self.timestamp = re.compile(r"^\s*" + cfg.timestamp.pattern + r"\s*$")
        for g in ("m", "s"):
            if f"(?P<{g}>" not in cfg.timestamp.pattern:
                raise ValueError(f"parser.yaml: timestamp.pattern needs a named group '{g}'")
        self.filename_episode = c(cfg.episode_number_from_filename, "episode_number_from_filename")
        self.header = {k: c(v, f"header.{k}") for k, v in cfg.header.model_dump().items()}
        self.script_start = c(cfg.script_start, "script_start")
        self.script_end = c(cfg.script_end, "script_end")
        self.scene = {k: c(v, f"scene.{k}") for k, v in cfg.scene.model_dump().items()}
        self.ignore = [c(p, "ignore") for p in cfg.ignore]
        self.block_header = c(cfg.block_header, "block_header")
        self.timing_line = c(cfg.timing_line, "timing_line")
        self.speaker_name = c(cfg.speaker.name, "speaker.name")
        self.parenthetical = c(cfg.speaker.parenthetical, "speaker.parenthetical")
        self.block_kinds = {k.strip().upper(): v for k, v in cfg.block_kinds.items()}
        self.voice_tags = {t.strip().upper().rstrip(".") for t in cfg.speaker.voice_tags}


# ---------------------------------------------------------------------------- loaders
def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def load_settings(path: str | Path = DEFAULT_SETTINGS) -> Settings:
    return Settings.model_validate(_read_yaml(Path(path)))


def apply_model_overrides(settings: Settings, overrides: dict[str, str] | None) -> Settings:
    """Copy of settings with {agent: "provider:model"} as each agent's main model (CLI --model,
    UI model pickers). The agent's configured backups stay behind it. Other parameters
    (temperature, kwargs) are kept only when the model is unchanged, since they're model-specific."""
    if not overrides:
        return settings
    s = settings.model_copy(deep=True)
    for agent, value in overrides.items():
        if ":" not in value:
            raise ValueError(f"--model {agent}={value}: use agent=provider:model")
        provider, model = value.split(":", 1)
        if provider not in s.providers:
            raise ValueError(f"--model {agent}={value}: unknown provider '{provider}' (have {sorted(s.providers)})")
        base = s.model_for(agent)
        same = (base.provider, base.model) == (provider, model)
        primary = base.model_copy(update={"fallbacks": []}) if same else ModelSpec(provider=provider, model=model)
        primary.fallbacks = [f for f in base.fallbacks if (f.provider, f.model) != (provider, model)]
        s.models[agent] = primary
    return s


def load_parser_config(path: str | Path | None = None, settings: Settings | None = None) -> ParserConfig:
    if path is None:
        settings = settings or load_settings()
        path = settings.resolve(settings.paths.parser_config)
    return ParserConfig.model_validate(_read_yaml(Path(path)))


def load_moment_types(path: str | Path | None = None, settings: Settings | None = None) -> MomentTypes:
    if path is None:
        settings = settings or load_settings()
        path = settings.resolve(settings.paths.moment_types)
    return MomentTypes.model_validate(_read_yaml(Path(path)))
