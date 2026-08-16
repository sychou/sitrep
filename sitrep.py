#!/usr/bin/env -S uv run --script --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openai",
#     "anthropic",
#     "pydantic",
# ]
# ///
"""Sitrep — a personalized daily briefing from the gumshoe archive.

Gumshoe gathers, sitrep reports. Sitrep reads what gumshoe filed since the last
report, works out the day's topics, cross-references them against Sean's own
writing (the Obsidian journal and notes, via qmd), and composes one synthesized
briefing: what the outside world said about things Sean is working on.

Four phases:
  1. Collect   — gumshoe archive items not yet covered (by stable item_id), plus
                 the recent daily journal for the relevance signal.
  2. Correlate — extract topics from the day's material, then query qmd
                 (obsidian + gumshoe collections) per topic for personal context.
  3. Compose   — summarize each new item (cached), then synthesize the brief.
  4. Record    — mark items covered in state, only after the report is on disk.

Config is yours (~/.config/sitrep/config.toml) and never written here; state is
the script's (~/.local/share/sitrep/state.json) and never hand-edited. The
gumshoe archive is read-only to sitrep. Model calls go to glm-5.2 on Ollama
Cloud via its OpenAI-compatible API; requires OLLAMA_API_KEY in the environment.

Usage:
    sitrep run [--label morning|evening] [--kindle] [--journal-only] [--dry-run]
    sitrep run --redo          # rebuild the last report from cached summaries
    sitrep status              # coverage, last run, uncovered-item preview
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import tomllib
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# ── Paths ───────────────────────────────────────────────────────────────────
VAULT = Path.home() / "Vaults" / "Main"
INBOX = VAULT / "Inbox"
JOURNAL_DIR = VAULT / "Daily"
GUMSHOE_VAULT = Path.home() / "Vaults" / "Gumshoe"

# Yours, never written by the script.
CONFIG_DIR = Path.home() / ".config" / "sitrep"
CONFIG_FILE = CONFIG_DIR / "config.toml"

# The script's, never hand-edited.
DATA_DIR = Path.home() / ".local" / "share" / "sitrep"
STATE_FILE = DATA_DIR / "state.json"
REPORTS_DIR = DATA_DIR / "reports"

OLLAMA_BASE_URL = "https://ollama.com/v1"
DEFAULT_MODEL = "glm-5.2"
MODEL = DEFAULT_MODEL  # display model; set per provider-run in do_run()

# Providers sitrep can synthesize with. glm goes through Ollama Cloud's
# OpenAI-compatible endpoint; claude goes through the native Anthropic SDK
# (never the OpenAI-compat shim). --models picks one or more by name.


@dataclass
class Provider:
    name: str              # registry key / short label for filenames ("glm", "openai")
    kind: str              # "openai" | "anthropic"
    model: str             # model id
    base_url: str = ""     # openai-compatible endpoints
    api_key_env: str = ""
    token_param: str = "max_tokens"   # gpt-5.x reasoning models want max_completion_tokens
    min_tokens: int = 0    # floor so reasoning/thinking leaves room for the JSON


PROVIDERS: dict[str, Provider] = {
    "glm": Provider("glm", "openai", "glm-5.2", OLLAMA_BASE_URL, "OLLAMA_API_KEY"),
    "openai": Provider("openai", "openai", "gpt-5.2", "https://api.openai.com/v1",
                       "OPENAI_API_KEY", token_param="max_completion_tokens", min_tokens=16_000),
    "gpt": Provider("openai", "openai", "gpt-5.2", "https://api.openai.com/v1",
                    "OPENAI_API_KEY", token_param="max_completion_tokens", min_tokens=16_000),
    "claude": Provider("claude", "anthropic", "claude-opus-5", min_tokens=32_000),
    "anthropic": Provider("claude", "anthropic", "claude-opus-5", min_tokens=32_000),
}

# See youscribe: Ollama Cloud does not enforce response_format, so a malformed
# reply is ordinary, and each retry costs only a call. Lean on retrying.
MODEL_ATTEMPTS = 3

JOURNAL_DAYS = 7          # how far back the journal signal reaches
MAX_TOPICS = 8            # cap on extracted topics that get a qmd query
QMD_HITS = 5              # correlation hits kept per topic per collection
SUMMARY_MAX_CHARS = 120_000
JOURNAL_MAX_CHARS = 4_000  # per journal entry, into the topic/brief prompts

SYSTEM_PROMPT = (
    "You are an expert analyst writing a personal intelligence briefing. "
    "Always respond in English. Respond with a single JSON object and nothing else."
)

# ── Prompts ─────────────────────────────────────────────────────────────────

SUMMARY_PROMPT = """\
Summarize the following item for a personal briefing. It is external content \
Sean has archived — a podcast/video transcript OR an email newsletter/article. \
The text may be in any language; always produce your output in English.

First decide what kind of item this is.

- "profile" — it is substantially ABOUT one person or one company: their story, \
background, how they built the thing, what the company does. Founder and \
executive interviews are profiles even when they range over industry topics.
- "news" — it is discussion, analysis or commentary on events, markets or \
releases; a panel show, a newsletter issue, a briefing.

If in doubt, ask what a reader would still want from this in a year. If the \
answer is "who that person or company is", it is a profile.

Provide:
1. episode_kind: "profile" or "news".
2. profile: when episode_kind is "profile", an object — subject (name), \
subject_type ("person" or "company"), role (title and company, or what the \
company does), detail (3-6 sentences on background and why they matter), and \
notable_facts (2-6 concrete facts: numbers, dates, funding, scale). When \
episode_kind is "news", set this to null.
3. notable_quotes: 1-3 of the most impactful direct quotes (verbatim, with \
speaker if identifiable). Max 3.
4. short_summary: one paragraph (3-5 sentences) capturing the main thesis.
5. key_takeaways: 4-8 bullet points of the most important insights or facts.
6. action_items: 0-5 concrete follow-ups a reader could take; empty if none.

Be specific and reference concrete details. Avoid generic filler.

Return ONLY a JSON object of exactly this shape — no prose, no markdown fences:

{{
  "episode_kind": "news" | "profile",
  "profile": null | {{"subject": "string", "subject_type": "person" | "company",
    "role": "string", "detail": "string", "notable_facts": ["string", ...]}},
  "notable_quotes": ["string", ...],
  "short_summary": "string",
  "key_takeaways": ["string", ...],
  "action_items": ["string", ...]
}}

Item — {source} ({source_type}), "{title}":
---
{body}
---
"""

TOPICS_PROMPT = """\
Below are summaries of {count} items the outside world produced recently, and \
excerpts from Sean's own daily journal (what he is currently working on and \
thinking about).

Extract the TOPICS that connect the external items to what Sean cares about — \
the things worth looking up in his personal knowledge base to find related \
notes, people, and projects. A topic is a short noun phrase (2-5 words): a \
company, a person, a technology, a market, a thesis. Prefer topics that appear \
in BOTH the external items and the journal, but include a purely-external topic \
if it is clearly significant. Return at most {max_topics}, most important first.

Return ONLY a JSON object of exactly this shape — no prose, no fences:

{{"topics": ["string", ...]}}

External item summaries:
---
{digest}
---

Recent journal:
---
{journal}
---
"""

BRIEF_PROMPT = """\
You are writing Sean's personal briefing. He will read only this, not the \
individual item summaries. Your job is to connect what the outside world said \
today to what Sean is actively working on, thinking about, or has history with \
— and to say plainly what does not connect to anything of his.

You are given: summaries of {count} external items filed since the last report; \
excerpts from Sean's recent daily journal; and, for each topic, related notes \
pulled from his own knowledge base (title and an excerpt). Today is {until}.

Write a synthesized brief, not a list. Weave the external items together and tie \
them to Sean's context. Lead with the two or three things that most intersect \
his active work. Where an item touches nothing in his journal or notes, do not \
force a connection — leave it out of the themes (it will be listed separately).

Provide:
1. headline: one paragraph (4-6 sentences) on the two or three developments that \
most intersect Sean's active work. Concrete, specific, personal — name the \
items and the personal context.
2. themes: 2-5 themes. For each: a title; a detail paragraph (3-6 sentences) \
that cites concrete specifics from the external items AND names the personal \
context that makes it relevant; sources (the exact external item titles it draws \
on); related (the exact note titles from the knowledge base it connects to, as \
given below — omit if none).
3. action_items: 0-8 consolidated, concrete follow-ups, deduplicated across \
items and the journal. Empty list if none.

Use exact item titles in `sources` and exact note titles in `related`. Be \
specific — cite numbers, names, claims.

Return ONLY a JSON object of exactly this shape — no prose, no fences:

{{
  "headline": "string",
  "themes": [{{"title": "string", "detail": "string",
    "sources": ["string", ...], "related": ["string", ...]}}, ...],
  "action_items": ["string", ...]
}}

External item summaries:
---
{digest}
---

Recent journal:
---
{journal}
---

Related personal notes by topic:
---
{correlations}
---
"""


# ── Schemas ─────────────────────────────────────────────────────────────────

class Profile(BaseModel):
    subject: str = Field(description="Name of the person or company profiled")
    subject_type: Literal["person", "company"] = Field(description="Which one it is")
    role: str = Field(description="Role, or what the company does")
    detail: str = Field(description="3-6 sentences: background, what they built, why")
    notable_facts: list[str] = Field(description="2-6 concrete facts")


class Summary(BaseModel):
    episode_kind: Literal["news", "profile"] = Field(
        description="'profile' if substantially about one person/company, else 'news'"
    )
    profile: Profile | None = Field(default=None, description="Present only when profile")
    notable_quotes: list[str] = Field(description="1-3 impactful verbatim quotes")
    short_summary: str = Field(description="A single paragraph (3-5 sentences)")
    key_takeaways: list[str] = Field(description="4-8 key insights or facts")
    action_items: list[str] = Field(description="0-5 concrete follow-ups; empty if none")


class Topics(BaseModel):
    topics: list[str] = Field(description="Short noun-phrase topics, most important first")


class Theme(BaseModel):
    title: str = Field(description="Short theme title")
    detail: str = Field(description="3-6 sentences tying external items to personal context")
    sources: list[str] = Field(description="Exact external item titles this draws on")
    related: list[str] = Field(default_factory=list, description="Exact personal note titles")


class Brief(BaseModel):
    headline: str = Field(description="One paragraph on the top intersections")
    themes: list[Theme] = Field(description="2-5 themes")
    action_items: list[str] = Field(description="0-8 consolidated follow-ups")


# ── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class Item:
    """One gumshoe archive item, read straight from its markdown file."""
    item_id: str
    source: str
    source_type: str        # "youtube" | "newsletter"
    title: str
    url: str
    published: str          # YYYY-MM-DD
    fetched: datetime
    path: Path
    body: str


@dataclass
class Entry:
    """One item summarized during this run, collected for the brief."""
    item: Item
    summary: Summary


@dataclass
class Hit:
    """One qmd correlation hit — a note from Sean's own knowledge base."""
    title: str
    note: str               # note name for a [[wikilink]]
    collection: str
    snippet: str


@dataclass
class Issue:
    scope: str
    detail: str


ISSUES: list[Issue] = []


@dataclass
class Usage:
    kind: str               # "summary" | "topics" | "brief"
    prompt: int
    completion: int
    retry: bool
    estimated: bool

    @property
    def total(self) -> int:
        return self.prompt + self.completion


USAGE: list[Usage] = []

# The summary phase runs once on a cheap model and is shared by every synthesis
# report; these carry its cost/issues into each report's own accounting.
SUMMARY_INFO: dict = {}          # {"model", "calls", "prompt", "completion"}
SUMMARY_ISSUES: list[Issue] = []


def record_usage(kind: str, prompt_tokens: int, completion_tokens: int,
                 prompt_text: str, reply: str, retry: bool) -> None:
    """Prefer the API's own token counts; fall back to ~4 chars/token."""
    if prompt_tokens:
        USAGE.append(Usage(kind, prompt_tokens, completion_tokens, retry, False))
    else:
        USAGE.append(Usage(kind, len(prompt_text) // 4, len(reply) // 4, retry, True))


def usage_totals() -> dict:
    stages: dict[str, dict] = {}
    for u in USAGE:
        row = stages.setdefault(u.kind, {"calls": 0, "prompt": 0, "completion": 0})
        row["calls"] += 1
        row["prompt"] += u.prompt
        row["completion"] += u.completion
    retries = [u for u in USAGE if u.retry]
    return {
        "stages": stages,
        "prompt": sum(u.prompt for u in USAGE),
        "completion": sum(u.completion for u in USAGE),
        "total": sum(u.total for u in USAGE),
        "calls": len(USAGE),
        "retries": len(retries),
        "retry_tokens": sum(u.total for u in retries),
        "estimated": any(u.estimated for u in USAGE),
    }


def record_issue(scope: str, detail: str) -> None:
    ISSUES.append(Issue(scope, detail))


# ── Config & state ──────────────────────────────────────────────────────────

def load_config() -> dict:
    """Optional. Missing config just means defaults and no Kindle delivery."""
    if not CONFIG_FILE.exists():
        return {}
    with CONFIG_FILE.open("rb") as fh:
        return tomllib.load(fh)


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(STATE_FILE)


def load_last_run() -> datetime | None:
    try:
        return datetime.fromisoformat(load_state()["last_run"])
    except (KeyError, ValueError):
        return None


def save_last_run(when: datetime) -> None:
    state = load_state()
    state["last_run"] = when.isoformat(timespec="seconds")
    save_state(state)


def reported_ids() -> dict[str, str]:
    """Item IDs a report has already covered, mapped to the report that did."""
    return load_state().get("reported", {})


def mark_reported(entries: list[Entry], report_name: str) -> int:
    state = load_state()
    reported = state.setdefault("reported", {})
    for e in entries:
        reported[e.item.item_id] = report_name
    save_state(state)
    return len(entries)


def unmark_reported(item_ids: set[str]) -> int:
    state = load_state()
    reported = state.get("reported", {})
    dropped = [i for i in item_ids if reported.pop(i, None) is not None]
    save_state(state)
    return len(dropped)


def cached_summary(model: str, item_id: str) -> Summary | None:
    raw = load_state().get("summaries", {}).get(model, {}).get(item_id)
    if not raw:
        return None
    try:
        return Summary.model_validate(raw)
    except ValidationError:
        return None


def store_summary(model: str, item_id: str, summary: Summary) -> None:
    state = load_state()
    state.setdefault("summaries", {}).setdefault(model, {})[item_id] = summary.model_dump()
    save_state(state)


# ── Text helpers (ported from youscribe) ────────────────────────────────────

FRONTMATTER_RE = re.compile(r"^([\w-]+):\s*(.*)$")


def safe_filename(text: str) -> str:
    return re.sub(r'[\\/:*?"<>|#^\[\]]', "", text).strip()


def slugify(text: str, limit: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(slug) > limit:
        slug = slug[:limit].rsplit("-", 1)[0]
    return slug or "untitled"


def yaml_scalar(text: str) -> str:
    return '"{}"'.format(text.replace("\\", "\\\\").replace('"', '\\"'))


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Return (frontmatter dict, body) for a gumshoe markdown file."""
    if not text.startswith("---\n") or (end := text.find("\n---\n", 4)) == -1:
        return {}, text
    front: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if m := FRONTMATTER_RE.match(line.strip()):
            value = m.group(2).strip()
            if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
                value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
            front[m.group(1)] = value
    return front, text[end + 5:].lstrip("\n")


# ── Phase 1: Collect ────────────────────────────────────────────────────────

def collect_items() -> list[Item]:
    """Every gumshoe archive item, read from its markdown. Oldest fetched first
    so the digest and windowing read chronologically."""
    items: list[Item] = []
    if not GUMSHOE_VAULT.is_dir():
        return items
    for path in sorted(GUMSHOE_VAULT.glob("*/*.md")):
        front, body = split_frontmatter(path.read_text(encoding="utf-8"))
        item_id = front.get("item_id")
        if not item_id:
            continue
        try:
            fetched = datetime.fromisoformat(front["fetched"])
        except (KeyError, ValueError):
            fetched = datetime.now(timezone.utc)
        items.append(Item(
            item_id=item_id,
            source=front.get("source", path.parent.name),
            source_type=front.get("source_type", "unknown"),
            title=front.get("title", path.stem),
            url=front.get("url", ""),
            published=front.get("published", ""),
            fetched=fetched,
            path=path,
            body=body,
        ))
    items.sort(key=lambda i: i.fetched)
    return items


def uncovered_items() -> list[Item]:
    covered = reported_ids()
    return [i for i in collect_items() if i.item_id not in covered]


def read_journal(days: int = JOURNAL_DAYS) -> list[tuple[date, str]]:
    """Recent daily journal entries, oldest first. Bounded to the year folders
    the window spans so it never walks two decades of Daily/."""
    today = datetime.now().astimezone().date()
    cutoff = today - timedelta(days=days)
    entries: list[tuple[date, str]] = []
    for year in sorted({cutoff.year, today.year}):
        ydir = JOURNAL_DIR / str(year)
        if not ydir.is_dir():
            continue
        for path in ydir.rglob("*.md"):
            m = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
            if not m:
                continue
            try:
                d = date.fromisoformat(m.group(1))
            except ValueError:
                continue
            if d < cutoff:
                continue
            entries.append((d, path.read_text(encoding="utf-8")))
    entries.sort(key=lambda e: e[0])
    return entries


def journal_text(entries: list[tuple[date, str]]) -> str:
    return "\n\n".join(
        f"Journal {d.isoformat()}:\n{text.strip()[:JOURNAL_MAX_CHARS]}"
        for d, text in entries
    ) or "(no recent journal entries)"


# ── Model calls (ported from youscribe) ─────────────────────────────────────

def extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start == -1:
        return text
    depth, in_string, escaped = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


class Caller:
    """One synthesizer, normalized across providers. complete() returns
    (text, prompt_tokens, completion_tokens, truncated) so call_model stays
    provider-agnostic."""

    def __init__(self, provider: Provider, config: dict):
        self.provider = provider
        if provider.kind == "openai":
            key = os.environ.get(provider.api_key_env) or config.get(provider.api_key_env.lower())
            if not key:
                raise SystemExit(f"{provider.api_key_env} not set for the {provider.name} provider.")
            self.client = OpenAI(api_key=key, base_url=provider.base_url)
        elif provider.kind == "anthropic":
            import anthropic
            key = os.environ.get("ANTHROPIC_API_KEY") or config.get("anthropic_api_key")
            if not key:
                raise SystemExit(
                    "Set ANTHROPIC_API_KEY (env) or anthropic_api_key in "
                    f"{CONFIG_FILE.name} to synthesize with the claude provider.")
            self.client = anthropic.Anthropic(api_key=key)
        else:
            raise SystemExit(f"unknown provider kind: {provider.kind}")

    def complete(self, system: str, prompt: str, max_tokens: int) -> tuple[str, int, int, bool]:
        if self.provider.kind == "openai":
            budget = {self.provider.token_param: max(max_tokens, self.provider.min_tokens)}
            r = self.client.chat.completions.create(
                model=self.provider.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                **budget,
            )
            ch = r.choices[0]
            u = getattr(r, "usage", None)
            return (ch.message.content or "",
                    getattr(u, "prompt_tokens", 0) or 0,
                    getattr(u, "completion_tokens", 0) or 0,
                    ch.finish_reason == "length")
        # anthropic — stream (opus-5 thinks by default; large max_tokens needs it)
        with self.client.messages.stream(
            model=self.provider.model,
            max_tokens=max(max_tokens, self.provider.min_tokens),
            system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            msg = stream.get_final_message()
        if msg.stop_reason == "refusal":
            raise RuntimeError("Claude declined the request (safety classifier)")
        text = "".join(b.text for b in msg.content if b.type == "text")
        u = msg.usage
        prompt_tokens = ((u.input_tokens or 0)
                         + (getattr(u, "cache_read_input_tokens", 0) or 0)
                         + (getattr(u, "cache_creation_input_tokens", 0) or 0))
        return text, prompt_tokens, u.output_tokens or 0, msg.stop_reason == "max_tokens"


def call_model(caller: Caller, prompt: str, schema: type[BaseModel],
               max_tokens: int = 8192, kind: str = "summary") -> BaseModel:
    """Ask the model for JSON matching `schema`, validated locally with up to
    MODEL_ATTEMPTS-1 corrective retries. A truncated reply retries at a larger
    budget; a malformed one replays the prompt with what was wrong."""
    model = caller.provider.model
    user = prompt
    problem = ""
    for attempt in range(1, MODEL_ATTEMPTS + 1):
        text, pt, ct, truncated = caller.complete(SYSTEM_PROMPT, user, max_tokens)
        record_usage(kind, pt, ct, SYSTEM_PROMPT + user, text, retry=attempt > 1)
        if truncated:
            problem = "the reply was cut off before the JSON object closed"
            max_tokens = int(max_tokens * 1.5)
        else:
            try:
                return schema.model_validate_json(extract_json(text))
            except (ValidationError, ValueError) as e:
                problem = str(e).splitlines()[0]
        if attempt < MODEL_ATTEMPTS:
            print(f"  unusable reply from {model} ({problem}); "
                  f"retrying [{attempt}/{MODEL_ATTEMPTS - 1}]")
            user = (prompt + f"\n\nYour previous reply could not be parsed: {problem}. "
                    "Reply again with ONLY the JSON object in the shape described above.")
    raise RuntimeError(
        f"{model} returned no usable JSON after {MODEL_ATTEMPTS} attempts: {problem}")


def summarize_item(caller: Caller, item: Item) -> Summary:
    """Summarize one item, cached per model so retries and --redo are free."""
    if cached := cached_summary(caller.provider.model, item.item_id):
        return cached
    summary = call_model(
        caller,
        SUMMARY_PROMPT.format(
            source=item.source, source_type=item.source_type,
            title=item.title, body=item.body[:SUMMARY_MAX_CHARS],
        ),
        Summary, kind="summary",
    )
    store_summary(caller.provider.model, item.item_id, summary)
    return summary


# ── Phase 2: Correlate ──────────────────────────────────────────────────────

def item_digest(entries: list[Entry]) -> str:
    """Per-item blocks for the topic and brief prompts."""
    blocks = []
    for e in entries:
        s = e.summary
        takeaways = "\n".join(f"  - {t}" for t in s.key_takeaways)
        kind = "PROFILE" if s.profile else "news"
        subj = f"Subject: {s.profile.subject} ({s.profile.role})\n" if s.profile else ""
        blocks.append(
            f"## {e.item.source} — {e.item.title}\n"
            f"Published: {e.item.published or '?'} · {e.item.source_type} · {kind}\n"
            f"{subj}{s.short_summary}\n"
            f"Key takeaways:\n{takeaways}\n"
        )
    return "\n".join(blocks)


def extract_topics(caller: Caller, entries: list[Entry], journal: str) -> list[str]:
    result = call_model(
        caller,
        TOPICS_PROMPT.format(
            count=len(entries), max_topics=MAX_TOPICS,
            digest=item_digest(entries), journal=journal,
        ),
        Topics, max_tokens=2048, kind="topics",
    )
    seen, topics = set(), []
    for t in result.topics:
        key = t.strip().lower()
        if key and key not in seen:
            seen.add(key)
            topics.append(t.strip())
    return topics[:MAX_TOPICS]


def qmd_query(topic: str, collection: str, n: int = QMD_HITS) -> list[Hit]:
    """Search one qmd collection for a topic. Failure is non-fatal — correlation
    is augmentation, not the spine."""
    try:
        proc = subprocess.run(
            ["qmd", "query", topic, "-c", collection, "--json", "-n", str(n)],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        record_issue("correlate", f"qmd {collection} '{topic}' — {type(e).__name__}")
        return []
    if proc.returncode != 0:
        return []
    try:
        rows = json.loads(proc.stdout) if proc.stdout.strip() else []
    except json.JSONDecodeError:
        return []
    hits = []
    for r in rows:
        file = r.get("file", "")
        note = Path(file).stem if file else (r.get("title") or "")
        snippet = " ".join((r.get("snippet") or "").split())[:280]
        hits.append(Hit(title=r.get("title") or note, note=note,
                        collection=collection, snippet=snippet))
    return hits


def correlate(topics: list[str]) -> dict[str, list[Hit]]:
    """Per topic, the personal notes it surfaces from the obsidian collection.
    Deduped by note within a topic; the gumshoe collection is skipped here since
    the day's items are already the external side."""
    out: dict[str, list[Hit]] = {}
    for topic in topics:
        seen, hits = set(), []
        for hit in qmd_query(topic, "obsidian"):
            if hit.note and hit.note not in seen:
                seen.add(hit.note)
                hits.append(hit)
        if hits:
            out[topic] = hits
    return out


def correlation_digest(correlations: dict[str, list[Hit]]) -> str:
    if not correlations:
        return "(no related personal notes found)"
    blocks = []
    for topic, hits in correlations.items():
        lines = "\n".join(f'  - "{h.title}": {h.snippet}' for h in hits)
        blocks.append(f"Topic: {topic}\n{lines}")
    return "\n".join(blocks)


# ── Phase 3: Compose ────────────────────────────────────────────────────────

def build_brief(caller: Caller, entries: list[Entry], journal: str,
                correlations: dict[str, list[Hit]], until: str) -> Brief:
    prompt = BRIEF_PROMPT.format(
        count=len(entries), until=until,
        digest=item_digest(entries), journal=journal,
        correlations=correlation_digest(correlations),
    )
    return call_model(caller, prompt, Brief, max_tokens=16384, kind="brief")


def featured_titles(brief: Brief) -> set[str]:
    return {s.strip().lower() for theme in brief.themes for s in theme.sources}


def is_featured(item: Item, featured: set[str]) -> bool:
    """Fuzzy match an item to a theme's cited sources — the model paraphrases."""
    t = item.title.strip().lower()
    return any(t in f or f in t for f in featured if f)


def report_path(label: str, date_str: str) -> Path:
    stem = f"Sitrep {label} {date_str}".strip().replace("  ", " ")
    path = REPORTS_DIR / f"{stem}.md"
    n = 2
    while path.exists():
        path = REPORTS_DIR / f"{stem} ({n}).md"
        n += 1
    return path


def url_map(entries: list[Entry]) -> dict[str, str]:
    return {e.item.title.strip().lower(): e.item.url for e in entries}


def link_sources(titles: list[str], urls: dict[str, str]) -> str:
    """Render a theme's external sources as links where we can resolve the URL."""
    out = []
    for t in titles:
        url = urls.get(t.strip().lower())
        out.append(f"[{t}]({url})" if url else t)
    return ", ".join(out)


def write_report(entries: list[Entry], brief: Brief, correlations: dict[str, list[Hit]],
                 since: str, until: str, label: str) -> Path:
    urls = url_map(entries)
    featured = featured_titles(brief)
    also = [e for e in entries if not is_featured(e.item, featured)]
    profiles = [e for e in entries if e.summary.profile]
    title = f"Sitrep — {label + ' ' if label else ''}{until}"

    lines = [
        "---",
        "kind:",
        "  - report",
        "report: sitrep",
        f"date: {until}",
        f"covering-from: {since}",
        *([f"label: {label}"] if label else []),
        f"items: {len(entries)}",
        f"profiles: {len(profiles)}",
        f"tokens: {usage_totals()['total']}",
        f"model: {MODEL}",
        f"issues: {len(ISSUES)}",
        "sources:",
        *[f"  - {yaml_scalar(s)}" for s in sorted({e.item.source for e in entries})],
        "---",
        "",
        f"# {title}",
        "",
        f"*{len(entries)} item(s) filed {since} → {until}, synthesized by "
        f"`{MODEL}`" + (f"; {len(ISSUES)} issue(s) — see Run Notes.*" if ISSUES else ".*"),
        "",
        brief.headline,
        "",
    ]

    if brief.themes:
        lines.extend(["## Themes", ""])
        for theme in brief.themes:
            lines.extend([f"### {theme.title}", "", theme.detail, ""])
            if theme.sources:
                lines.append(f"*External: {link_sources(theme.sources, urls)}*")
            if theme.related:
                lines.append(f"*Related: {', '.join(f'[[{n}]]' for n in theme.related)}*")
            if theme.sources or theme.related:
                lines.append("")

    if brief.action_items:
        lines.extend(["## Action Items", ""])
        lines.extend(f"- {a}" for a in brief.action_items)
        lines.append("")

    if profiles:
        lines.extend(["## Profiles", ""])
        for e in sorted(profiles, key=lambda x: x.summary.profile.subject.lower()):
            p = e.summary.profile
            lines.extend([f"### {p.subject} — {p.role}", "", p.detail, ""])
            if p.notable_facts:
                lines.extend(f"- {f}" for f in p.notable_facts)
                lines.append("")
            lines.extend([
                f"*{p.subject_type.title()} · {e.item.source}, {e.item.published}"
                + (f" · [source]({e.item.url})*" if e.item.url else "*"), ""])

    if also:
        lines.extend(["## Also Filed", ""])
        for e in sorted(also, key=lambda x: x.item.source):
            link = f" — [source]({e.item.url})" if e.item.url else ""
            lines.append(f"- **{e.item.source}** · {e.item.title}{link}")
        lines.append("")

    lines.extend(_usage_section())

    issues = SUMMARY_ISSUES + ISSUES
    if issues:
        lines.extend([
            "## Run Notes", "",
            (f"{len(issues)} item(s) had trouble this run. Items are left uncovered "
             "on failure, so anything here is retried next run."), ""])
        lines.extend(f"- **{i.scope}** — {i.detail}" for i in issues)
        lines.append("")

    path = report_path(label, until)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _usage_section() -> list[str]:
    use = usage_totals()
    # Fold in the shared summary phase (run once on the cheap model) so each
    # report's total reflects the true cost of producing it.
    calls = use["calls"] + SUMMARY_INFO.get("calls", 0)
    prompt = use["prompt"] + SUMMARY_INFO.get("prompt", 0)
    completion = use["completion"] + SUMMARY_INFO.get("completion", 0)
    if not calls:
        return []
    label = {"topics": "Topic extraction", "brief": "Brief synthesis"}
    lines = [
        "## Token Usage", "",
        "| Stage | Calls | Prompt | Completion | Total |",
        "| ----- | ----: | -----: | ---------: | ----: |",
    ]
    if SUMMARY_INFO.get("calls"):
        s = SUMMARY_INFO
        lines.append(f"| Item summaries ({s['model']}, shared) | {s['calls']} | "
                     f"{s['prompt']:,} | {s['completion']:,} | {s['prompt'] + s['completion']:,} |")
    for kind in ("topics", "brief"):
        if row := use["stages"].get(kind):
            lines.append(f"| {label[kind]} | {row['calls']} | {row['prompt']:,} | "
                         f"{row['completion']:,} | {row['prompt'] + row['completion']:,} |")
    lines.append(f"| **Total** | **{calls}** | **{prompt:,}** | "
                 f"**{completion:,}** | **{prompt + completion:,}** |")
    lines.append("")
    notes = []
    if use["retries"]:
        plural = "" if use["retries"] == 1 else "s"
        notes.append(f"{use['retries']} retry call{plural} cost an extra "
                     f"{use['retry_tokens']:,} tokens")
    if use["estimated"]:
        notes.append("some counts estimated from length; the API returned none")
    if notes:
        lines.extend([f"*{'. '.join(notes).capitalize()}.*", ""])
    return lines


# ── EPUB + Kindle (ported from youscribe) ───────────────────────────────────

EPUB_CSS = """\
body { margin: 0 1em; }
h1 { font-size: 1.4em; margin: 1em 0 0.2em; }
h2 { font-size: 1.15em; margin: 1.4em 0 0.3em; }
h3 { font-size: 1em; margin: 1.2em 0 0.3em; }
.meta { font-style: italic; margin: 0 0 1.2em; }
.sources { font-style: italic; font-size: 0.9em; }
blockquote { margin: 0.8em 1.2em; font-style: italic; }
"""


def xhtml_page(title: str, body: str, nav: bool = False) -> str:
    epub_ns = ' xmlns:epub="http://www.idpf.org/2007/ops"' if nav else ""
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'
        f'<html xmlns="http://www.w3.org/1999/xhtml"{epub_ns} xml:lang="en" lang="en">\n'
        f'<head><meta charset="utf-8"/><title>{esc(title)}</title>'
        '<link rel="stylesheet" type="text/css" href="style.css"/></head>\n'
        f"<body>\n{body}\n</body>\n</html>\n"
    )


def brief_xhtml(entries: list[Entry], brief: Brief, since: str, until: str, label: str) -> str:
    urls = url_map(entries)
    featured = featured_titles(brief)
    parts = [
        f"<h1>Sitrep — {esc((label + ' ') if label else '')}{esc(until)}</h1>",
        (f'<p class="meta">{len(entries)} item(s) filed {esc(since)} &#8594; {esc(until)}, '
         f"synthesized by {esc(MODEL)}.</p>"),
        f"<p>{esc(brief.headline)}</p>",
    ]
    if brief.themes:
        parts.append("<h2>Themes</h2>")
        for theme in brief.themes:
            parts.append(f"<h3>{esc(theme.title)}</h3><p>{esc(theme.detail)}</p>")
            bits = []
            if theme.sources:
                srcs = ", ".join(
                    f'<a href="{esc(urls[s.strip().lower()])}">{esc(s)}</a>'
                    if urls.get(s.strip().lower()) else esc(s) for s in theme.sources)
                bits.append(f"External: {srcs}")
            if theme.related:
                bits.append("Related: " + ", ".join(esc(n) for n in theme.related))
            if bits:
                parts.append(f'<p class="sources">{" · ".join(bits)}</p>')
    if brief.action_items:
        parts.append("<h2>Action Items</h2><ul>")
        parts.extend(f"<li>{esc(a)}</li>" for a in brief.action_items)
        parts.append("</ul>")
    also = [e for e in entries if not is_featured(e.item, featured)]
    if also:
        parts.append("<h2>Also Filed</h2><ul>")
        parts.extend(
            f'<li><strong>{esc(e.item.source)}</strong> · '
            f'<a href="{esc(e.item.url)}">{esc(e.item.title)}</a></li>'
            if e.item.url else
            f"<li><strong>{esc(e.item.source)}</strong> · {esc(e.item.title)}</li>"
            for e in also)
        parts.append("</ul>")
    if ISSUES:
        parts.append("<h2>Run Notes</h2><ul>")
        parts.extend(f"<li><strong>{esc(i.scope)}</strong> — {esc(i.detail)}</li>" for i in ISSUES)
        parts.append("</ul>")
    return xhtml_page(f"Sitrep {until}", "\n".join(parts))


def item_xhtml(entry: Entry) -> str:
    s, item = entry.summary, entry.item
    parts = [
        f"<h1>{esc(item.title)}</h1>",
        f'<p class="meta">{esc(item.source)} · {esc(item.source_type)} · {esc(item.published)}</p>',
    ]
    if p := s.profile:
        parts.append(f"<h2>Profile: {esc(p.subject)}</h2>")
        parts.append(f'<p class="meta">{esc(p.subject_type.title())} · {esc(p.role)}</p>')
        parts.append(f"<p>{esc(p.detail)}</p>")
        if p.notable_facts:
            parts.append("<ul>" + "".join(f"<li>{esc(f)}</li>" for f in p.notable_facts) + "</ul>")
    if s.notable_quotes:
        parts.append("<h2>Quotes</h2>")
        parts.extend(f"<blockquote><p>{esc(q.strip().lstrip('>').strip())}</p></blockquote>"
                     for q in s.notable_quotes[:3])
    parts.append(f"<h2>Summary</h2><p>{esc(s.short_summary)}</p>")
    parts.append("<h2>Key Takeaways</h2><ul>")
    parts.extend(f"<li>{esc(t)}</li>" for t in s.key_takeaways[:8])
    parts.append("</ul>")
    if s.action_items:
        parts.append("<h2>Action Items</h2><ul>")
        parts.extend(f"<li>{esc(a)}</li>" for a in s.action_items)
        parts.append("</ul>")
    if item.url:
        parts.append(f'<p><a href="{esc(item.url)}">Source</a></p>')
    return xhtml_page(item.title, "\n".join(parts))


def build_epub(entries: list[Entry], brief: Brief, since: str, until: str,
               label: str, path: Path) -> Path:
    order = sorted(entries, key=lambda x: (x.item.source, x.item.published))
    docs = [("report.xhtml", f"Sitrep {until}", brief_xhtml(order, brief, since, until, label))]
    for i, e in enumerate(order, 1):
        docs.append((f"item-{i}.xhtml", f"{e.item.source} — {e.item.title}", item_xhtml(e)))

    manifest = "\n".join(
        f'    <item id="d{i}" href="{name}" media-type="application/xhtml+xml"/>'
        for i, (name, _, _) in enumerate(docs))
    spine = "\n".join(f'    <itemref idref="d{i}"/>' for i in range(len(docs)))
    modified = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    opf = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="bookid">urn:sitrep:{esc(path.stem)}</dc:identifier>
    <dc:title>{esc(path.stem)}</dc:title>
    <dc:creator>Sitrep</dc:creator>
    <dc:language>en</dc:language>
    <dc:date>{esc(until)}</dc:date>
    <meta property="dcterms:modified">{modified}</meta>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="css" href="style.css" media-type="text/css"/>
{manifest}
  </manifest>
  <spine>
{spine}
  </spine>
</package>
"""
    toc = "\n".join(f'<li><a href="{name}">{esc(title)}</a></li>' for name, title, _ in docs)
    nav = xhtml_page("Contents",
                     f'<nav epub:type="toc" id="toc"><h1>Contents</h1>\n<ol>\n{toc}\n</ol></nav>',
                     nav=True)
    container = """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip",
                   compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", container)
        z.writestr("OEBPS/content.opf", opf)
        z.writestr("OEBPS/nav.xhtml", nav)
        z.writestr("OEBPS/style.css", EPUB_CSS)
        for name, _, contents in docs:
            z.writestr(f"OEBPS/{name}", contents)
    return path


def send_to_kindle(path: Path, address: str, sender: str, until: str) -> None:
    subprocess.run(
        ["gog", "gmail", "send", "--account", sender, "--to", address,
         "--subject", f"Sitrep {until}", "--body", "Sent by sitrep.",
         "--attach", str(path)],
        check=True, capture_output=True, text=True,
    )


# ── Pipeline ────────────────────────────────────────────────────────────────

def summarize_all(caller: Caller, items: list[Item]) -> list[Entry]:
    entries: list[Entry] = []
    for item in items:
        print(f"  {item.published or '?'} | {item.source} — {item.title}")
        try:
            entries.append(Entry(item, summarize_item(caller, item)))
        except Exception as e:  # noqa: BLE001
            print(f"    error summarizing: {e}")
            record_issue(f"{item.source} — {item.title}", f"summarize failed — {str(e)[:160]}")
    return entries


def resolve_provider(name: str) -> Provider:
    id_to_name = {p.model: p.name for p in PROVIDERS.values()}
    key = name if name in PROVIDERS else id_to_name.get(name)
    if not key:
        sys.exit(f"unknown model/provider {name!r}; available: {', '.join(sorted(PROVIDERS))}")
    return PROVIDERS[key]


def compose_one(caller: Caller, entries: list[Entry], journal: str,
                since: str, until: str, label: str, config: dict,
                kindle: bool) -> Path | None:
    """Correlate + synthesize + write for one synthesis model over shared,
    pre-computed summaries. Only the synthesis model's tokens are spent here;
    the shared summary cost is folded into the report via SUMMARY_INFO."""
    global MODEL
    MODEL = caller.provider.model
    USAGE.clear()
    ISSUES.clear()
    tag = f"[{caller.provider.name}] "

    print(f"{tag}── correlate: extracting topics ({MODEL})")
    try:
        topics = extract_topics(caller, entries, journal)
    except Exception as e:  # noqa: BLE001
        print(f"{tag}  topic extraction failed ({e}); proceeding without correlation")
        record_issue("correlate", f"topic extraction failed — {str(e)[:160]}")
        topics = []
    if topics:
        print(f"{tag}  topics: {', '.join(topics)}")
    correlations = correlate(topics)
    print(f"{tag}  {sum(len(v) for v in correlations.values())} related note(s) "
          f"across {len(correlations)} topic(s)")

    print(f"{tag}── compose: synthesizing the brief")
    try:
        brief = build_brief(caller, entries, journal, correlations, until)
        path = write_report(entries, brief, correlations, since, until, label)
    except Exception as e:  # noqa: BLE001
        print(f"{tag}report FAILED ({e}); items stay uncovered")
        return None

    vault_copy = INBOX / path.name
    if INBOX.is_dir():
        shutil.copyfile(path, vault_copy)
    use = usage_totals()
    synth = use["total"]
    print(f"{tag}   report: {path}")
    if INBOX.is_dir():
        print(f"{tag}   vault:  {vault_copy}")
    print(f"{tag}done: {len(entries)} item(s); {synth:,} synthesis tokens "
          f"(+{SUMMARY_INFO.get('prompt', 0) + SUMMARY_INFO.get('completion', 0):,} shared summaries)")

    if kindle:
        address, sender = config.get("kindle"), config.get("kindle_sender")
        if not address or not sender:
            missing = "kindle" if not address else "kindle_sender"
            print(f"{tag}   kindle: no '{missing} = ...' line in {CONFIG_FILE.name}")
        else:
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    epub = build_epub(entries, brief, since, until, label,
                                      Path(tmp) / f"{path.stem}.epub")
                    send_to_kindle(epub, address, sender, path.stem)
                print(f"{tag}   kindle: sent to {address}")
            except subprocess.CalledProcessError as e:
                print(f"{tag}   kindle: send failed — {(e.stderr or '').strip()[:200]}")
            except Exception as e:  # noqa: BLE001
                print(f"{tag}   kindle: send failed — {type(e).__name__}: {e}")
    return path


def do_run(providers: list[Provider], config: dict, args) -> int:
    started = datetime.now().astimezone()

    redo_ids: set[str] | None = None
    if args.redo:
        last = load_state().get("last_report")
        if not last or not last.get("items"):
            sys.exit("No previous report recorded in state.json — nothing to redo.")
        redo_ids = set(last["items"])
        unmark_reported(redo_ids)
        print(f"reopened {len(redo_ids)} item(s) from {last.get('name', 'the last report')}")

    items = uncovered_items()
    if redo_ids is not None:
        # --redo rebuilds exactly the last report — not the whole backlog.
        items = [i for i in items if i.item_id in redo_ids]
    total = len(items)
    if args.limit and total > args.limit:
        items = items[-args.limit:]
        print(f"── collect: {len(items)} of {total} uncovered item(s) (--limit {args.limit})")
    else:
        print(f"── collect: {total} uncovered item(s) in the gumshoe archive")
    if not items and not args.journal_only:
        print("done: nothing new since the last report — no report written "
              "(use --journal-only to brief over the journal alone)")
        return 0

    journal = journal_text(read_journal())

    if args.dry_run:
        print(f"── would run {[p.name for p in providers]} over {len(items)} item(s)")
        for item in items:
            print(f"  {item.published or '?'} | {item.source} — {item.title}")
        print(f"── journal: {len(read_journal())} recent entr(y/ies) in reach")
        return 0

    last_run = load_last_run()
    until = started.strftime("%Y-%m-%d")
    earliest = min((i.published for i in items if i.published), default="")
    since = (min(last_run.strftime("%Y-%m-%d"), earliest or until) if last_run
             else earliest or until)

    # ── Summarize once on the cheap model; every synthesis report shares it. ──
    global SUMMARY_INFO, SUMMARY_ISSUES
    summary_provider = resolve_provider(config.get("summary_model", "glm"))
    USAGE.clear()
    ISSUES.clear()
    print(f"── summarize: {len(items)} item(s) with {summary_provider.model} (shared)")
    entries = summarize_all(Caller(summary_provider, config), items)
    if not entries:
        print(f"done: none of {len(items)} item(s) summarized — no report")
        return 1
    su = usage_totals()["stages"].get("summary", {"calls": 0, "prompt": 0, "completion": 0})
    SUMMARY_INFO = {"model": summary_provider.model, **su}
    SUMMARY_ISSUES = list(ISSUES)

    multi = len(providers) > 1
    written: list[Path] = []
    for provider in providers:
        # Tag the filename by model only when comparing several. A single
        # synthesis model is THE report (glm summaries + this model's brief),
        # not a per-model variant, so it's just "Sitrep <date>".
        parts = [args.label] + ([provider.name] if multi else [])
        base = " ".join(x for x in parts if x)
        path = compose_one(Caller(provider, config), entries, journal,
                           since, until, base, config, args.kindle)
        if path:
            written.append(path)

    if not written:
        print("\ndone: no reports written; items stay uncovered and are retried")
        return 1

    # Commit once: items are covered regardless of which model reported them.
    save_last_run(started)
    state = load_state()
    state["last_report"] = {"name": written[-1].name,
                            "items": [e.item.item_id for e in entries]}
    save_state(state)
    marked = mark_reported(entries, written[-1].name)
    print(f"\n── record: {marked} item(s) marked covered across {len(written)} report(s)")
    return 0


def do_status() -> int:
    state = load_state()
    print(f"state: {STATE_FILE}")
    print(f"last run: {state.get('last_run', 'never')}")
    reported = state.get("reported", {})
    print(f"covered: {len(reported)} item(s)")
    last = state.get("last_report")
    if last:
        print(f"last report: {last.get('name')} ({len(last.get('items', []))} items)")
    pending = uncovered_items()
    print(f"\nuncovered: {len(pending)} item(s)")
    for item in pending[:12]:
        print(f"  {item.published or '?':10}  {item.source_type:10}  "
              f"{item.source} — {item.title[:60]}")
    if len(pending) > 12:
        print(f"  ... and {len(pending) - 12} more")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Personalized daily briefing from the gumshoe archive.")
    sub = parser.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="collect, correlate, compose, record")
    run.add_argument("--label", default="", help="report label, e.g. morning or evening")
    run.add_argument("--models", default="",
                     help="comma-separated providers to run, one report each "
                          f"(available: {', '.join(sorted(PROVIDERS))}; default: config or glm)")
    run.add_argument("--kindle", action="store_true", help="also email each report as an EPUB to the Kindle")
    run.add_argument("--journal-only", action="store_true", help="brief over the journal even with no new items")
    run.add_argument("--redo", action="store_true", help="rebuild the last report from cached summaries")
    run.add_argument("--dry-run", action="store_true", help="collect + preview only; no model calls or writes")
    run.add_argument("--limit", type=int, default=None,
                     help="cover only the N most-recently-fetched uncovered items (bounds a big first run)")

    sub.add_parser("status", help="coverage, last run, uncovered items")
    args = parser.parse_args()

    if args.cmd == "status":
        sys.exit(do_status())
    if args.cmd != "run":
        parser.print_help()
        sys.exit(0)

    config = load_config()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if cm := config.get("claude_model"):  # e.g. "claude-sonnet-5"
        PROVIDERS["claude"].model = PROVIDERS["anthropic"].model = cm
    if om := config.get("openai_model"):  # e.g. "gpt-5.4"
        PROVIDERS["openai"].model = PROVIDERS["gpt"].model = om

    # Resolve the provider list: --models, else config, else glm.
    names = ([n.strip() for n in args.models.split(",") if n.strip()]
             or config.get("models") or [config.get("model", "glm")])
    # Config may carry model IDs (e.g. "glm-5.2"); map those to their provider name.
    id_to_name = {p.model: p.name for p in PROVIDERS.values()}
    providers: list[Provider] = []
    seen = set()
    for n in names:
        key = n if n in PROVIDERS else id_to_name.get(n)
        if not key or key in seen:
            if not key:
                sys.exit(f"unknown model/provider {n!r}; available: {', '.join(sorted(PROVIDERS))}")
            continue
        seen.add(key)
        providers.append(PROVIDERS[key])

    sys.exit(do_run(providers, config, args))


if __name__ == "__main__":
    main()
