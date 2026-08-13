# Sitrep — Requirements

Status: draft for review, 2026-08-07

Sitrep writes the daily report. It reads what gumshoe gathered, works out
what is news today, cross-references it against Sean's own writing and
history, and produces one personally relevant briefing: what the outside
world said about things Sean is actively working on, thinking about, or has
history with.

Gumshoe gathers, sitrep reports. The two share one interface — the gumshoe
archive and its qmd index — and nothing else. Sitrep is the successor to
youscribe's reporting half (the rollup, EPUB, and Kindle delivery); the
generic cross-show rollup becomes a personalized brief.

## What "news for the day" means

Sitrep determines the day's material by date across several corpora:

- The gumshoe archive is the anchor. It carries the most news-driven data
  points — transcripts, articles, newsletters filed since the last report —
  and every run starts from what is new there.
- The daily journal says what Sean is currently working on and thinking
  about. It is the primary signal for relevance, not a source of news.
- Email and Slack augment: recent messages can confirm that a topic is live
  (a thread about a vendor, a channel discussing an announcement) and can
  contribute items the archive lacks.

Everything else in the PKM — people notes, project notes, reference — is not
scanned by date at all. It is queried by topic, via qmd, once the day's
topics are known.

## The report

One markdown document per run. Working structure, inherited in part from
youscribe and expected to evolve:

- a headline paragraph: the two or three things today that most intersect
  with Sean's active work
- themes: each connecting external items to the personal context that makes
  them relevant ("All-In covered X; you wrote about X in Tuesday's journal;
  your note on Y from March is related"), with links to sources on both
  sides — URLs for external items, vault paths for personal context
- profiles: episodes or articles substantially about one person or company
  get a structured record (subject, role, background, concrete facts),
  cross-referenced against existing people/company notes in the vault
- consolidated action items, deduplicated across sources
- run notes: what failed, what was skipped, what is deferred — a partial
  report must never read as a complete one
- token usage and cost, as in youscribe

The report is written to a reports directory under sitrep's own data dir and
copied into the vault Inbox for reading; the vault copy is disposable.
`--kindle` packages it as an EPUB and mails it to the Kindle, carrying the
full per-item detail as chapters — the mechanism (stdlib zipfile, `gog gmail
send`, approved-sender list) ports from youscribe unchanged.

## Pipeline

A run has four phases:

1. Collect — enumerate new items in the gumshoe archive since the last
   report; pull the recent journal entries; optionally sweep recent email
   and Slack for augmenting signals.
2. Correlate — extract the topics from the day's material, then query qmd
   (`obsidian` and `gumshoe` collections) per topic for the personal context:
   related notes, prior journal mentions, people and project pages.
3. Compose — summarize each new item, then synthesize the report from the
   item summaries plus the correlation results. Model calls follow
   youscribe's discipline: JSON shape requested in the prompt and validated
   locally, corrective retries, token budget growth on truncation.
4. Record — mark the covered items in state, only after the report is on
   disk. A run that fails before writing records nothing and the next run
   re-covers it.

Like youscribe, coverage is tracked in sitrep's own `state.json` as a map of
item ID to the report that covered it. Gumshoe's archive is read-only to
sitrep; sitrep never writes outside its own data dir, the vault Inbox, and
outbound mail.

## Layout

```text
~/.config/sitrep/config.toml       # kindle addresses, model, source toggles
~/.local/share/sitrep/
    state.json                     # coverage map, last run
    reports/Sitrep <date>.md       # every report ever written
```

## Interfaces

- Gumshoe archive: read the files directly (frontmatter carries source,
  date, URL, stable ID) and search them via the `gumshoe` qmd collection.
- Vault: journal and notes via the `obsidian` qmd collection; direct file
  reads for the most recent journal entries.
- Email: via msgvault (recent-message queries), not Gmail directly —
  msgvault is the system of record and already handles identity resolution.
- Slack: mechanism open — see open questions.
- Model: an OpenAI-compatible API, as in youscribe. Which model is an open
  question.

## Behavior notes

- Runs are proactive: scheduled daily, after gumshoe's run and after fresh
  archive items are embedded, so semantic correlation sees today's content.
  Manual runs behave identically.
- A day with no new archive items produces no report by default (nothing is
  news), unless invoked with a flag to report over journal activity alone.
- Failures are per-item where possible: an item that fails to summarize is
  reported in run notes, stays uncovered, and is retried next run at no cost.
- `--redo` rebuilds the last report after a prompt or model change by
  reopening its items in state, then taking the ordinary path.

## Open questions

- Model choice. Youscribe used `glm-5.2` on Ollama Cloud. The correlate and
  compose phases are more demanding than per-episode summarization —
  synthesis quality is the whole product — which argues for a stronger model
  (Claude via API?) at higher cost. Possibly split: cheap model for item
  summaries, strong model for the brief.
- Topic extraction. What is a "topic"? Extracted by the model from item
  summaries and journal entries, then used as qmd queries? Or a maintained
  interest list in config that the day's material is matched against? The
  first is zero-maintenance, the second is steerable.
- Slack access. The claude.ai Slack MCP is interactive; a scheduled script
  needs its own token. Is Slack worth a v1 slot, or a later augmentation?
- Email sweep scope. Which msgvault queries define "recent email that
  signals a live topic" without dragging in the whole inbox?
- Relevance threshold. When a new item touches nothing in the journal or
  PKM, does it still get a line in the report (youscribe-style coverage), or
  is it dropped? Dropping is what makes the brief personal; covering
  everything is what makes it complete. Possibly a short "also filed" list.
- Cadence. Daily is the model, but gumshoe may run more often. Should sitrep
  ever produce more than one report a day, or always sweep everything
  uncovered into the next morning's brief?
- Journal privacy. Journal text flows into model prompts. Fine for a cloud
  model, or a reason to keep the correlate phase local?
