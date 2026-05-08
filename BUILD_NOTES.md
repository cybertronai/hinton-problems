# Session Report: Building hinton-problems via Agent Teams

**Session ID:** `d8af4bb0-1435-4528-a5da-ac91c30b7bcb`
**Project:** SutroYaro (the lead session was checked out there)
**Output:** [cybertronai/hinton-problems](https://github.com/cybertronai/hinton-problems) — 53 stubs, all merged
**Span:** 2026-05-01 21:52 → 2026-05-04 03:35 (~30 wall hours, with overnight idle gaps)
**Source:** the full jsonl is at `~/.claude/projects/-Users-yadkonrad-dev-dev-year26-feb26-SutroYaro/d8af4bb0-...jsonl` (5.1 MB, 3,033 events)

This report is what the session log actually shows, suitable for a team video.

---

## TL;DR for the video opener

- 53 Hinton-paper stubs implemented in 30 wall hours, ~800k tokens, on Claude Opus 4.7 with the 1M-token context window.
- The **SPEC was a single GitHub issue** ([#1](https://github.com/cybertronai/hinton-problems/issues/1)).
- The **dispatcher was Claude Code's `agent-teams` primitive** — one team, ten waves, fresh teammates per wave.
- **One human prompt of intent ("use parallel team of agents… DONT USE THE SKILL CRAP!")** turned a serial workflow into a 10-wave parallel build.
- All work routed through GitHub: 18 issues, 15 PRs, audits via subagents, merges only on user approval.

---

## The actual chain of events

| Time (UTC) | Event |
|---|---|
| 05-01 21:52 | Session opens in SutroYaro |
| 05-01 21:53 | Yad invokes the `sutro-sync` skill — the only skill call in the session — to pull Telegram chat, Google Docs, and GitHub context |
| 05-01 21:59 | Yad: *"lets focus on hinton, pull it into may26, ok, shall we pull hinto-porblems, make a branch and then try doing SPEC, branch, and then github issues / what do we think?"* — the SPEC-first idea is born |
| 05-01 22:04 | Yad: *"Don't merge anything, we need to pull it, we need to open up a GitHub issue and create the spec as a GitHub issue saying that's what you will follow"* — SPEC = issue |
| 05-01 22:13 | **Issue #1 opened**: *Spec: minimum implementation requirements for stub problems (v1)*. Authored by `agent-0bserver07 (Claude Code) on behalf of Yad`. Lists required files, 8-section README template, reproducibility rules, acceptance checklist. |
| 05-02 05:21 | Yad: *"ok do u see Yaroslav's comment, can that help with pre-context for our waves of agents?"* — Yaroslav had commented on issue #1 overnight |
| 05-02 05:48 | Yad: *"deploy all the waves one after another given Yaroslav's comment and our spec and the local repo of hintons problems, and do branches per waves"* |
| 05-02 05:51 | Yad: ***"I need you to use parallel team of agents that claude code has built in, DONT USE THE SKILL CRAP! https://code.claude.com/docs/en/agent-teams"*** |
| 05-02 05:51 | Lead dispatches a `claude-code-guide` subagent to read the agent-teams docs |
| 05-02 05:53 | **`TeamCreate`** — team `hinton-impl` born. agent_type `orchestrator`. Description: *"Each teammate owns one stub, works in its own worktree at `/may26/hinton-problems-waves/`, pushes branch `impl/<slug>`, opens PR. Lead is the SutroYaro session; reviews PRs and merges only on user approval."* |
| 05-02 05:55 → 06:07 | **Wave 0**: single-stub spike. `xor-builder` teammate spawned, builds, opens PR #3. Sanity check passes. |
| 05-02 06:07 → 09:20 | **Wave 1**: 3 teammates (`n-bit-parity-builder`, `symmetry-builder`, `negation-builder`). All three open PRs. Then shut down via `SendMessage(shutdown_request)`. |
| 05-02 09:21 → 13:46 | **Wave 2**: 5 teammates (binary-addition, encoder-3-parity, encoder-4-3-4, encoder-8-3-8, encoder-backprop-8-3-8). |
| 05-02 13:49 → 15:37 | **Wave 3**: 6 teammates (encoder-40-10-40, shifter, grapheme-sememe, distributed-to-local-bottleneck, t-c-discrimination, recurrent-shift-register). |
| 05-02 14:48 | Yad: *"why are there so many PRs? Weren't there supposed to be 5 waves?"* — turning point. From here, **multiple stubs per PR**, one PR per wave. |
| 05-02 15:37 → 20:17 | **Waves 4 → 7**: 6 stubs each. PR titles read like a tour: *tier-B 1980s-90s classics*, *Helmholtz/MDL/Imax/fast-weights*, *TRBM/RTRBM/gated-RBM/RNN/factorial-VQ/eGLOM*, *MNIST cluster (FF + distillation + capsule precursor)*. |
| 05-02 21:10 | **Wave 8**: 6 stubs (external-data + harder architectures). |
| 05-03 04:20 | **Wave 9**: 5 stubs. 50/53 v1 done. |
| 05-03 22:56 | **Wave 10**: final 3 stubs (AIR + matrix capsules). **v1 complete at 53/53.** |
| 05-03 23:18 → 23:55 | Docs PRs: RESULTS.md, MkDocs site, **switch to mdBook (after Yad pushed back hard)**, 4-column catalog tables. |
| 05-04 00:34 | Introduction page in mdBook style + Unlicense. |
| 05-04 00:55 | **v1 gap analysis** issue opened — umbrella tracker for 25 partials + 1 non-replication. |
| 05-04 01:08 | Yad: *"so whats left, coz we aending this sessions 800k"* — context budget watermark. |
| 05-04 03:35 | Last event in the session log. |

---

## The SPEC (issue #1) — the actual contract

The contract between Yad and every teammate was a single GitHub issue. Not chat. Not a system prompt. An issue every PR linked back to.

It defined:
- **Required files** per stub: `<name>.py`, `README.md`, `make_<name>_gif.py`, `visualize_<name>.py`, `<name>.gif`, `viz/`
- **8 README sections**: Header / Problem / Files / Running / Results / Visualizations / Deviations / Open questions
- **Reproducibility rules**: seed exposed via CLI, all hyperparameters in Results, command in §Running reproduces the number
- **Acceptance checklist** (8 checkboxes): reproduces under 5 min on a laptop / final accuracy with seed / GIF / weight viz / training curves / deviations section / open questions / no `NotImplementedError`
- **Out of scope for v1**: energy metric (deferred to v2 ByteDMD), GPU / large-scale runs

That's the entire DSL. Every stub had to fit.

---

## The orchestration model

```
                     ┌──────────────────┐
                     │ hinton-impl team │  (TeamCreate, agent_type=orchestrator)
                     └─────────┬────────┘
                               │
                  ┌────────────┼────────────┐
                  │            │            │
            Wave 0/1/2…    SendMessage   Subagent dispatches
                               │            │
                               ▼            ▼
                          ┌──────────┐  ┌──────────────┐
                          │ teammates │  │ Agent tool   │
                          │ <slug>-   │  │ (general-    │
                          │ builder   │  │  purpose,    │
                          │ x53       │  │  Explore)    │
                          └────┬─────┘  └──────┬───────┘
                               │               │
                               ▼               ▼
                       worktree branch    PR audits, code reads
                       impl/<slug>
                               │
                               ▼
                          gh pr create
                               │
                               ▼
                          PR review + merge (Yad approves)
                               │
                               ▼
                       SendMessage(shutdown_request)
                               │
                               ▼
                          Next wave starts fresh
```

**Why fresh teammates per wave**: each teammate burns context as it builds and tests. Shutting down between waves keeps later waves running on full context windows. The lead persists; the workers turn over.

---

## What the session actually used

### Tool calls (in the lead session)

| Tool | Calls | What for |
|---|---|---|
| Bash | 191 | git, gh CLI, file ops, running tests |
| Read | 124 | reading paper PDFs, stub code, READMEs |
| Agent | **62** | subagent dispatches (see breakdown below) |
| Write | 55 | new files (READMEs, scripts, configs) |
| **SendMessage** | **53** | inter-teammate messaging (mostly wave shutdowns) |
| TaskUpdate | 24 | shared task list maintenance |
| TaskCreate | 22 | new tasks added to the team's list |
| Edit | 10 | small in-place edits |
| ToolSearch | 3 | loading deferred tool schemas |
| WebFetch | 2 | external doc reads |
| **Skill** | **1** | only `sutro-sync` at session start |
| **TeamCreate** | **1** | the `hinton-impl` team itself |

### Subagent dispatches (Agent tool, n=62)

| Type | Count | Use |
|---|---|---|
| `general-purpose` | 54 | per-stub builders ("Build xor stub for hinton-problems") |
| `Explore` | 7 | PR audits, stub correctness checks, wave reviews |
| `claude-code-guide` | 1 | researched the agent-teams docs at session start |

### GitHub artifacts produced

- **18 issues created** (1 SPEC + 15 per-stub issues for early waves + 2 follow-up: v2 ByteDMD, v1 gap analysis)
- **15 PRs created** (10 wave PRs + 5 docs PRs)
- **6 PRs merged via `gh pr merge`** in-session (the rest were merged separately by Yad)
- **24 git pushes**

### Token consumption — measured from JSONL session logs

The harness display the lead session was showing during the build (something like `~k/1M (% used)`) is **the current context window utilisation, not cumulative tokens consumed**. It answers "how much room is left in the 1M-token window?", not "how much did the build cost?". The honest cost number requires aggregating the JSONL files for the lead + every subagent.

Counted across the 63 JSONL session files in `~/.claude/projects/-Users-yadkonrad-dev-dev-year26-feb26-SutroYaro/` within the build window (2026-05-01T21:00 → 2026-05-04T23:30 UTC):

| Bucket | Tokens | % of total |
|---|---:|---:|
| Input (uncached, fresh content sent to the model) | 381,505 | 0.06% |
| Output (model generations) | 8,248,370 | 1.25% |
| Cache creation (first-time write of a prefix into the cache) | 34,376,850 | 5.20% |
| **Cache read** (re-loading already-cached prefix on subsequent turns) | **617,889,626** | **93.49%** |
| **Total tokens touched** | **660,896,351** | 100% |

About **661 million tokens** crossed the model boundary during this build. Why cache reads dominate: 1,069 lead-session assistant turns × growing conversation history × Anthropic's prompt caching means each turn re-reads the system prompt + tool definitions + prior turns out of cache (heavy discount) instead of paying full input rate.

63 distinct sessions worth of work participated: lead + 62 subagent dispatches (54 builders + 7 Explore auditors + 1 claude-code-guide). Claude Code spawns each subagent dispatch in its own session; the lead's JSONL only records the dispatch call and the subagent's final return, not the subagent's internal turns.

The full explainer of how to read these numbers (and how the harness UI display ≠ build cost) is in [issue #56](https://github.com/cybertronai/hinton-problems/issues/56). Companion to [schmidhuber-problems #19](https://github.com/cybertronai/schmidhuber-problems/issues/19) — same correction, same machinery.

### Skills

- **One skill call.** `sutro-sync`, used once at the very start to pull Telegram + Google Docs + GitHub context.
- After that, **Yad explicitly told the lead to use `agent-teams` instead of skills**: *"DONT USE THE SKILL CRAP!"*

That's the cleanest single data point about which mechanism is right for what:
- Skill = a recipe for "do this set of steps once at start"
- Agent team = parallel workers with shared task list
- They serve different purposes; this session used both, just sparingly.

---

## The waves at a glance

| Wave | Stubs | Highlights |
|---|---|---|
| 0 | 1 | xor (sanity check, single teammate) |
| 1 | 3 | n-bit-parity, symmetry, negation |
| 2 | 5 | binary-addition + encoder family |
| 3 | 6 | tier-B 1980s foundational (shifter, grapheme-sememe, etc.) |
| 4 | 6 | tier-B 1980s-90s classics |
| 5 | 6 | Helmholtz / MDL / Imax / fast-weights |
| 6 | 6 | TRBM / RTRBM / gated-RBM / RNN / factorial-VQ / eGLOM |
| 7 | 6 | MNIST cluster: FF + distillation + capsule precursor |
| 8 | 6 | external-data + harder architectures |
| 9 | 5 | final hard stubs (50/53 v1 done) |
| 10 | 3 | AIR + matrix capsules — **v1 complete at 53/53** |

Total: **53 stubs in 11 waves**.

---

## Yad's interaction pattern (the human side)

70 typed prompts across 30 wall hours. Most of them were one of three types:

**Type A — high-leverage direction (rare, big effects):**
- *"shall we pull hinton-problems, make a branch and then try doing SPEC, branch, and then github issues"* — chose the SPEC-as-issue model
- *"deploy multiple parallel workstreams to get things done under our supervision, by doing waves"* — chose the wave model
- *"I need you to use parallel team of agents that claude code has built in, DONT USE THE SKILL CRAP!"* — chose agent-teams
- *"why are there so many PRs? Weren't there supposed to be 5 waves?"* — collapsed per-stub PRs into per-wave PRs
- *"why didnt u use mdbook like i asked?"* — pushed back when the lead drifted (MkDocs got swapped for mdBook)

**Type B — status checks (frequent, low cost):**
- *"status?"* / *"status, what is left?"* / *"whats up"* — appears ~10 times. The lead summarises and continues.

**Type C — review and merge approvals:**
- *"i dont wanna merge yet, lets do audits left and then finish the implementation other waves right?"*
- *"ok wana comment on the PR that are ready to merge with evidence?"*
- *"Should we get up an issue with the context of the partials and the no?"* → led to the v1 gap analysis issue

The session also has frustrated moments. They are part of an honest report: when the lead drifted on docs tooling, Yad swore at it, and the lead course-corrected within minutes. Worth showing in a team video as the realistic version of "human in the loop."

---

## What this session actually proves

1. **A SPEC issue is enough contract.** No instruction file, no role prompt — just a versioned GitHub issue every PR points at. Acceptance checklist becomes the PR review template.
2. **Waves with fresh teammates beat one long-running team.** The lead persists; workers turn over per wave. This is what kept the run inside 800k tokens.
3. **`agent-teams` is the dispatcher; subagents are the workers.** This session used both: `TeamCreate` to spin up the team, then `Agent(subagent_type=general-purpose)` 54 times to actually build the stubs. Each teammate's work happened inside its subagent.
4. **Audits via separate Explore subagents.** 7 of the 62 Agent dispatches were reviewers, not builders. Keeps review context separate from build context.
5. **GitHub is the substrate.** Issues created the work, PRs delivered it, comments coordinated with Yaroslav, merges gated on Yad. No Slack. No call.
6. **One human per session is enough.** 70 prompts, mostly status checks. 5–6 of them set direction. The rest let the lead run.

---

## Concrete numbers you can quote in the video

- **53 / 53** Hinton-paper stubs implemented
- **27 reproduce** paper claims, **25 partial** (gap documented), **1 honest non-replication**
- **~30 wall hours**, with overnight idle gaps
- **63 distinct sessions** (lead + 62 subagent dispatches) consuming **~661 million tokens total**, of which **93.49% is cache_read** (re-loaded prefix from prior turns). Harness "~800k" display was current context-window utilisation, not cumulative cost. Full breakdown in [issue #56](https://github.com/cybertronai/hinton-problems/issues/56).
- **1 GitHub issue** as the SPEC
- **1 `TeamCreate`**, **53 named teammates**, **11 waves**
- **18 issues + 15 PRs** filed
- **62 subagent dispatches** (54 builders + 7 auditors + 1 docs-research)
- **191 bash, 124 reads, 55 writes, 53 SendMessages, 1 Skill**
- **70 human prompts** total; ~6 set direction, ~10 were status checks, ~10 were merge approvals, the rest were follow-ups
- **6 PRs merged in-session via `gh pr merge`**, 24 pushes

---

## Suggested video shot list

1. **Open on the SPEC issue** ([#1](https://github.com/cybertronai/hinton-problems/issues/1)) on screen. *"This is the entire contract."*
2. **Cut to the GitHub PRs page** filtered to "wave" — show the 10 wave PRs. *"This is what came out of it."*
3. **Show the agent-teams docs page** ([code.claude.com/docs/en/agent-teams](https://code.claude.com/docs/en/agent-teams)). *"This is the primitive that made parallel cheap."*
4. **Show the `TeamCreate` JSON** (in this report). *"One call. One team."*
5. **Walk through one wave** — pick wave 5 (Helmholtz/MDL/Imax/fast-weights, 6 stubs). Show the 6 teammate names, the 6 PRs, the merged commit.
6. **Show a single per-stub README** (e.g., `encoder-4-2-4`) — show how it satisfies all 8 spec sections.
7. **Show the v1 gap analysis issue** at the end. *"v1 = correctness. v1.5 = paper parity. v2 = energy."*
8. **Close on the bottom-line numbers** (53 / 30 hr / 800k / 1 spec / 11 waves).

---

*Generated from the live session log on 2026-05-04. Throwaway artifact — delete after the video is recorded.*
