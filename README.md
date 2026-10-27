# Hermes Living Profile Memory Provider

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Hermes Agent](https://img.shields.io/badge/Hermes_Agent-Compatible-orange.svg)](https://github.com/NousResearch/hermes-agent)

A high-performance, reusable, and open-source **Memory Provider Plugin** for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Combines the best attributes of **Anthropic's living Markdown cognitive files** (`PROFILE.md`), **OpenAI's automatic fact distillation**, and **full SQLite conversation archiving with FTS5 BM25 search**.

---

## Why This Plugin?

Default agent memory systems suffer from two extremes:
1. **Minimalist / Passive:** Default memory files (`MEMORY.md`, `USER.md`) only update if the model explicitly remembers to call a tool, leaving 95% of project decisions unindexed.
2. **Naive RAG / Write Amplification:** Many custom memory services drop and re-embed entire user histories on every turn ($O(N)$ write amplification), blowing up latency and burning API budgets.

**Living Profile** solves this with:
- **Full Turn Archiving:** Every conversation turn is indexed into local SQLite in <2ms.
- **$O(1)$ Markdown AST Diffing:** Facts are compiled into structured Markdown sections (`## GTM Strategy`, `## Tech Stack`). Only modified sections trigger updates.
- **Sub-5ms FTS5 Recall:** Multilingual BM25 full-text keyword matching retrieves relevant context before turns begin.
- **Fork & Branch Deduplication:** Chat branching (`turn_hash` + `parent_session_id`) skips re-indexing ancestor messages.
- **Privacy & Memory Control:** Real-time commands to pause recording or purge distracting sessions.

---

## Architecture & Turn Lifecycle

```
User Turn
   │
   ├──> 1. prefetch(query) [Sub-5ms]
   │       Searches PROFILE.md sections & conversation_turns via FTS5.
   │       Injects relevant context block without invalidating prompt cache.
   │
   ├──> 2. Model Generates Response
   │
   └──> 3. sync_turn() [Asynchronous]
           ├── Logs turn to SQLite & FTS5 (synchronous, <2ms)
           └── Heuristic background thread extracts key conventions to PROFILE.md
```

---

## Installation & Setup

### 1. Install into Hermes
Clone or copy this folder into your Hermes user plugins directory:

```bash
mkdir -p ~/.hermes/plugins/
git clone https://github.com/useing123/hermes-living-profile.git ~/.hermes/plugins/living_profile
```

### 2. Activate in Hermes Config
Enable it as your active memory provider in `~/.hermes/config.yaml`:

```bash
hermes config set memory.provider living_profile
```

Verify that Hermes discovers and activates the plugin:
```bash
hermes memory status
```
Output:
```text
  Provider:  living_profile
  Plugin:    installed ✓
  Status:    available ✓ (active)
```

---

## Features & Agent Tools

The plugin registers 4 native agent tools:

| Tool | Action | Description |
| :--- | :--- | :--- |
| `profile_view` | Read | Inspects the living `PROFILE.md` file or a specific heading. |
| `profile_search` | Search | Runs fast FTS5 BM25 keyword search across all past conversations. |
| `profile_update` | Write | Explicitly updates or appends a durable fact to a section. |
| `memory_control` | Control | `pause` recording, `resume` recording, or `purge` (wipe) current session memories. |

### Memory Control & Privacy Commands
In chat, you can instruct the agent:
- *"Do not record memory from this conversation"* $\rightarrow$ Invokes `memory_control(action='pause')`.
- *"Resume memory recording"* $\rightarrow$ Invokes `memory_control(action='resume')`.
- *"Delete all memories from this chat"* $\rightarrow$ Invokes `memory_control(action='purge')`.

---

## File Layout

- `plugin.yaml`: Hermes plugin manifest and capabilities declaration.
- `__init__.py`: Implements `MemoryProvider` ABC lifecycle hooks (`prefetch`, `sync_turn`, `on_session_switch`).
- `storage.py`: SQLite engine, FTS5 virtual tables, and Markdown AST parser.
- `config_schema.py`: Declarative web UI / config schema.

---

## License

MIT License. Open-source and free for the Hermes Agent and AI developer community.
