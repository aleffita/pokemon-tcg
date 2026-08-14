# Cross-Harness & Tokenizer Adaptation Guide

**Document**: Architecture Specification & Multi-Agent Portability Guide  
**Author**: Research Director & Alefita (Fitalabs AI Research)  
**Target Ingestion**: GPT-5.6 Sol, DeepSeek-V4-Pro, Codex, Claude 3.7 Sonnet / Flash, Open-Source Agentic Runtimes  
**Date**: August 14, 2026  
**Classification**: Protocol Portability & Epistemic Translation Specification  

---

## 1. Executive Purpose & Context

This document establishes the **Universal Portability Layer** for the Pokémon TCG AI research repository. 

While the foundational monographs, the Metanoia suite, and the 139k-match dataset were constructed using Google Antigravity 2.0 and Gemini 3.7 Flash High, the upcoming research phases and final submission execution may operate on **different agentic harnesses** (e.g., Claude Code, OpenAI Codex CLI, Cursor, Aider, OpenHands, AutoGen, LangGraph) and **different model tokenizers** (Claude 3.7, GPT-5.6 Sol, DeepSeek-V4-Pro).

This guide provides the exact semantic translation matrix, tokenizer control token mappings, and an exhaustive filesystem index of all session transcripts, subagent trees, and architectural blueprints.

---

## 2. Tokenizer-Aware Channels vs. Cross-Model Adaptation Matrix

The Gemini 3.7 Flash High and Gemma 4 model architectures leverage dedicated vocabulary tokens for channel isolation. When migrating to other frontier model families, the harness must adapt the prompting format according to the model's native reasoning paradigm:

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                                  CROSS-MODEL REASONING ADAPTATION                                 |
|                                                                                                   |
|  [Google / Gemini 3.7 & Gemma 4]                                                                  |
|  - Native Tokens: <|turn>, <turn|>, <|channel|>, <thought>, <|tool_call|>, <|tool_response|>      |
|  - State Machine: INIT -> GENERATE -> DEBATE -> RANK -> EVOLVE -> META_REVIEW                     |
|                                                                                                   |
|  [Anthropic / Claude 3.7 Sonnet & Flash]                                                          |
|  - Native Tag: <antml:thought> or extended thinking tags (<thinking> ... </thinking>)             |
|  - Adaptation: Inject Channel Protocol states inside native thinking blocks before tool output.  |
|                                                                                                   |
|  [DeepSeek / DeepSeek-V4-Pro & R1]                                                                |
|  - Native Tag: <think> ... </think>                                                               |
|  - Adaptation: Embed the 6-state dialectical loop directly into the autoregressive <think> stream |
|                                                                                                   |
|  [OpenAI / GPT-5.6 Sol & Codex]                                                                   |
|  - Native Architecture: Hidden reasoning tokens / CoT encapsulation                               |
|  - Adaptation: Enforce dialectical debate in system prompt; emit clean markdown directly to user. |
+───────────────────────────────────────────────────────────────────────────────────────────────────+
```

### 2.1. Exact Translation Protocol for New Models

If the incoming model does not have special tokenizer tokens for `<|channel|>`, it **MUST NOT** hallucinate invalid vocabulary IDs. Instead, it should adopt the following universal formatting rule:

```markdown
<!-- REASONING STATE: [INIT|GENERATE|DEBATE|RANK|EVOLVE|HALT|RESPOND] (Iteration N) -->
[Internal dialectical synthesis and trade-off debate]
<!-- END REASONING -->

[High-density technical response rendered in GitHub-flavored Markdown]
```

---

## 3. Harness Abstraction Translation Matrix

The Antigravity harness tools map directly to equivalent concepts in other multi-agent frameworks:

| Antigravity Tool / Primitive | Functionality | Equivalent in Claude Code / Codex / CLI | Equivalent in LangGraph / AutoGen |
| :--- | :--- | :--- | :--- |
| `invoke_subagent` / `define_subagent` | Spawns background worker agent | Sub-process / Worktree invocation (`git worktree`) | `WorkerNode` / `ConversableAgent` |
| `manage_subagents(kill/kill_all)` | Terminates active subagents | `kill -9` / Process group termination | Graph state cancellation / `terminate()` |
| `view_file` | Reads slice of file with byte offset | `head`, `tail`, `cat`, or native file reader | `ReadFileTool` |
| `replace_file_content` | Contiguous block replacement | `sed`, `patch`, or LLM diff applicator | `EditFileTool` |
| `multi_replace_file_content` | Non-contiguous chunk replacements | Multi-hunk patch application | `MultiHunkEditor` |
| `run_command` (with `WaitMsBeforeAsync`) | Async/sync shell execution | Terminal command execution (`bash`, `zsh`) | `BashTool` / `ExecTool` |
| `manage_task(list/kill/send_input)` | Inspects / controls background tasks | `ps`, `pkill`, `tmux`, `screen` | Process manager thread pool |
| `schedule` (timer / cron) | Dispatches delayed or recurring notification | `sleep`, `cron`, system timers | Scheduled trigger nodes |

---

## 4. Master Filesystem Directory & URI Index

Incoming models on any harness can directly inspect the persistent artifacts, conversation JSONL logs, and multi-agent survey states using the exact local filesystem paths indexed below:

### 4.1. Antigravity Brain Transcripts & Conversation Logs
* **Root App Data Directory**: `/Users/alefita/.gemini/antigravity-cli/`
* **Main Session Artifacts & Workspace**:  
  `/Users/alefita/.gemini/antigravity-cli/brain/9189fa2e-93c2-4a04-9bf0-6d090880de27/`
* **Main Session Compact Transcript (JSONL)**:  
  `/Users/alefita/.gemini/antigravity-cli/brain/9189fa2e-93c2-4a04-9bf0-6d090880de27/.system_generated/logs/transcript.jsonl`
* **Main Session Untruncated Raw Transcript (Full JSONL)**:  
  `/Users/alefita/.gemini/antigravity-cli/brain/9189fa2e-93c2-4a04-9bf0-6d090880de27/.system_generated/logs/transcript_full.jsonl`
* **Task Execution Logs Directory**:  
  `/Users/alefita/.gemini/antigravity-cli/brain/9189fa2e-93c2-4a04-9bf0-6d090880de27/.system_generated/tasks/`
* **Scratch Probes & Diagnostic Scripts**:  
  `/Users/alefita/.gemini/antigravity-cli/brain/9189fa2e-93c2-4a04-9bf0-6d090880de27/scratch/`

### 4.2. Specialized Subagent Conversation Transcripts
Each spawned subagent has an independent conversation log on disk:
1. **Deck Analyst Subagent** (`687d75b1-19ff-4999-a56e-c5d40d869f0f`):  
   `/Users/alefita/.gemini/antigravity-cli/brain/687d75b1-19ff-4999-a56e-c5d40d869f0f/.system_generated/logs/transcript.jsonl`
2. **Migration Engineer Subagent** (`245904ec-c5c7-4dca-a796-e04db0afb510`):  
   `/Users/alefita/.gemini/antigravity-cli/brain/245904ec-c5c7-4dca-a796-e04db0afb510/.system_generated/logs/transcript.jsonl`
3. **Teamwork Swarm Sentinel** (`da74ca6c-c155-4353-82d1-5c5695a60da1`):  
   `/Users/alefita/.gemini/antigravity-cli/brain/da74ca6c-c155-4353-82d1-5c5695a60da1/.system_generated/logs/transcript.jsonl`
4. **Teamwork Swarm Orchestrator** (`cd851a4f-6875-4819-9f25-1b23dd14cc1b`):  
   `/Users/alefita/.gemini/antigravity-cli/brain/cd851a4f-6875-4819-9f25-1b23dd14cc1b/.system_generated/logs/transcript.jsonl`

### 4.3. Workspace Multi-Agent Survey Memory (`.agents/`)
Sealed in Git commit `f988eb0`:
* **Original Request**: [`.agents/ORIGINAL_REQUEST.md`](file:///Users/alefita/workdir/pokemon-tcg/.agents/ORIGINAL_REQUEST.md)
* **Sentinel Memory**: [`.agents/sentinel_1/BRIEFING.md`](file:///Users/alefita/workdir/pokemon-tcg/.agents/sentinel_1/BRIEFING.md)
* **Orchestrator Memory**: [`.agents/orchestrator_1/BRIEFING.md`](file:///Users/alefita/workdir/pokemon-tcg/.agents/orchestrator_1/BRIEFING.md), [`DISPATCH.md`](file:///Users/alefita/workdir/pokemon-tcg/.agents/orchestrator_1/DISPATCH.md), [`progress.md`](file:///Users/alefita/workdir/pokemon-tcg/.agents/orchestrator_1/progress.md)
* **Milestone 1 Track (Neural Architecture)**:
  * [`.agents/sub_orch_m1/`](file:///Users/alefita/workdir/pokemon-tcg/.agents/sub_orch_m1/) (`BRIEFING.md`, `SCOPE.md`, `GATE_STATUS.md`, `progress.md`)
  * [`.agents/m1_exp_contract/`](file:///Users/alefita/workdir/pokemon-tcg/.agents/m1_exp_contract/), [`.agents/m1_exp_moe/`](file:///Users/alefita/workdir/pokemon-tcg/.agents/m1_exp_moe/), [`.agents/m1_exp_ropend/`](file:///Users/alefita/workdir/pokemon-tcg/.agents/m1_exp_ropend/), [`.agents/m1_worker_1/`](file:///Users/alefita/workdir/pokemon-tcg/.agents/m1_worker_1/)
* **Milestone 2 Track (Dataset & Oracles)**:
  * [`.agents/sub_orch_m2/`](file:///Users/alefita/workdir/pokemon-tcg/.agents/sub_orch_m2/) (`BRIEFING.md`, `SCOPE.md`, `progress.md`)
  * [`.agents/explorer_m2_dataset/`](file:///Users/alefita/workdir/pokemon-tcg/.agents/explorer_m2_dataset/), [`.agents/explorer_m2_db/`](file:///Users/alefita/workdir/pokemon-tcg/.agents/explorer_m2_db/), [`.agents/explorer_m2_oracle/`](file:///Users/alefita/workdir/pokemon-tcg/.agents/explorer_m2_oracle/)
* **Milestone 3 Track (Mathematical Monograph & PageRank)**:
  * [`.agents/sub_orch_m3/`](file:///Users/alefita/workdir/pokemon-tcg/.agents/sub_orch_m3/) (`BRIEFING.md`, `SCOPE.md`, `GATE_STATUS.md`, `progress.md`)
  * [`.agents/sub_orch_m3/explorer_1/`](file:///Users/alefita/workdir/pokemon-tcg/.agents/sub_orch_m3/explorer_1/), `explorer_2/`, `explorer_3/`
* **Test Writers & Harness Engineering**:
  * [`.agents/test_writer_tier1/`](file:///Users/alefita/workdir/pokemon-tcg/.agents/test_writer_tier1/), `tier2/`, `tier3/`, `tier4/`, [`.agents/orch_e2e/`](file:///Users/alefita/workdir/pokemon-tcg/.agents/orch_e2e/)
* **Survey Explorers**:
  * [`.agents/survey_explorer_1/`](file:///Users/alefita/workdir/pokemon-tcg/.agents/survey_explorer_1/), `survey_explorer_2/`, `survey_explorer_3/`

### 4.4. Preserved Master Blueprints
* **Execution Plan**: [`PROJECT.md`](file:///Users/alefita/workdir/pokemon-tcg/PROJECT.md)
* **Testing Infrastructure**: [`TEST_INFRA.md`](file:///Users/alefita/workdir/pokemon-tcg/TEST_INFRA.md)
* **Teamwork Prompt Draft**: [`prompt_draft.md`](file:///Users/alefita/.gemini/antigravity-cli/brain/9189fa2e-93c2-4a04-9bf0-6d090880de27/prompt_draft.md)
* **Master Implementation Plan**: [`master_project_state_and_handoff_plan.md`](file:///Users/alefita/.gemini/antigravity-cli/brain/9189fa2e-93c2-4a04-9bf0-6d090880de27/master_project_state_and_handoff_plan.md)

### 4.5. Canonical Knowledge & Memory
* **Living Research Memory**: [`GEMINI.md`](file:///Users/alefita/workdir/pokemon-tcg/GEMINI.md)
* **Wikifita Canonical Repository**: `/Users/alefita/Claude/wikifita/`
* **Wikifita Next.js Site**: `/Users/alefita/Claude/wikifita-site/`
* **Wikifita Skill**: [`.agents/skills/wikifita/SKILL.md`](file:///Users/alefita/workdir/pokemon-tcg/.agents/skills/wikifita/SKILL.md)

---

## 5. Guidance for Incoming Frontier Models

1. **Do not re-explore from scratch**: Consult `PROJECT.md` and `docs/technical_handoff_rfc.md` first. All database relationships, tensor shapes, and math proofs have been validated.
2. **Preserve ASD-STE100 & KaTeX Rules**: Render KaTeX math exclusively on standalone display blocks (`$$ ... $$`), never inside Markdown headings or bold text.
3. **Focus on the August 16 Deadline**: Transition immediately to Phase 2 (MoE training or GRPO policy refinement) and execute 500-match validation using `scripts/tournament.py`.
