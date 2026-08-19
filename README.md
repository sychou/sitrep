# Sitrep

Personalized intelligence briefs from a [gumshoe](https://github.com/sychou/gumshoe) archive.

Gumshoe gathers, sitrep reports. Sitrep reads what gumshoe filed since the
last report, works out the topics, optionally cross-references them against
the reader's own writing and archives, and composes one or more synthesized
briefings — what the outside world said about the things the reader cares
about. The two tools share one interface: the archive's markdown frontmatter.

## Briefs

Every report is a configured `[brief.*]` section claiming a subset of archive
sources. Each brief gets its own report file, its own coverage tracking, and
its own options:

- `title` — filename and display label
- `contextualizers` — which personal-context sources to use (omitted = all
  configured; `[]` = none)
- `focus` — a standing angle steering topic extraction and synthesis
- `audience` — who the brief is for; setting it switches from a personal
  briefing addressed to you into a third-person analyst brief for that
  readership
- `kindle` / `kindle_sender` — EPUB delivery for this brief when run with
  `--kindle`; briefs without them are skipped

Briefs are explicit-only: at least one must be configured, sources no brief
claims are never reported (runs warn about them), and a source listed in two
briefs appears in both.

## Contextualizers

Personal context reaches the pipeline only through configured
`[contextualizer.*]` sections, each with a type:

- `journal` — a folder of dated markdown files; contributes an ambient
  digest to the topic and synthesis prompts
- `qmd` — local markdown note search; contributes per-topic correlation
  lookups
- `msgvault` — email/meeting archive search; per-topic lookups over a
  recency window

Nothing is implicit. A brief with no contextualizers skips topic extraction
and correlation entirely and synthesizes from the item summaries alone.

## Setup

```bash
mkdir -p ~/.config/sitrep
cp config.example.toml ~/.config/sitrep/config.toml   # then edit
```

`config.example.toml` documents every key: models, paths, contextualizers,
and briefs. Model providers need their keys in the environment —
`OLLAMA_API_KEY` for glm on Ollama Cloud, `OPENAI_API_KEY` for OpenAI,
`ANTHROPIC_API_KEY` for Claude.

## Usage

```bash
sitrep run                    # run every configured brief
sitrep run --brief work       # one or more briefs, comma-separated
sitrep run --dry-run          # preview the per-brief partition; no model calls
sitrep run --limit 10         # bound a big first run
sitrep run --kindle           # also EPUB-to-Kindle the briefs configured for it
sitrep run --redo             # rebuild each brief's last report from cached summaries
sitrep run --models glm,openai  # compare synthesis models, one report each
sitrep status                 # per-brief coverage, last report, uncovered preview
```

Run via `uv` — single self-contained script with a PEP 723 header, no venv,
no install step:

```bash
./sitrep.py run
```

## How it works

Four phases, per brief:

1. **Collect** — archive items in the brief's sources not yet covered (by
   stable item ID), plus ambient context from its contextualizers.
2. **Correlate** — extract topics from the day's material, then query each
   lookup contextualizer per topic for related personal notes and messages.
3. **Compose** — summarize each new item on a cheap model (cached per model,
   so an item shared by two briefs is summarized once), then synthesize the
   brief on the configured model. JSON shape is requested in the prompt and
   validated locally, with corrective retries and budget growth on
   truncation.
4. **Record** — mark items covered, only after the report is on disk. A run
   that fails before writing records nothing; the next run retries.

Reports carry their own accounting: sources, token usage per stage, and run
notes for anything that failed. A day with no new items produces no report.

## Layout

```text
~/.config/sitrep/config.toml            # yours; never written by the script
~/.local/share/sitrep/
    state.json                          # per-brief coverage, summary cache
    reports/Sitrep <brief> <date>.md    # every report ever written
```

Reports are also copied into the folder named by `[paths] inbox_copy` when
one is configured (for example an Obsidian vault inbox); that copy is
disposable. The gumshoe archive is read-only to sitrep.
