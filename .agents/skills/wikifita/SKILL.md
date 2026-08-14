---
name: wikifita
description: Consult and maintain Wikifita, Alefita's persistent knowledge base and memory at ~/Claude/wikifita. Use whenever the user mentions Wikifita, memory, profile, preferences, directives, people, contacts, previous decisions, cross-project context, or editorial workflow; also when creating, updating, consolidating, indexing, or auditing wiki content. Also triggers on artifact creation (applies visual identity from directives). MEMORY.md is the master memory index. CLAUDE.md is the canonical instruction file and AGENTS.md must be its compatible symlink.
---

# Wikifita

Treat `~/Claude/wikifita/` as the canonical, versioned, shared source of knowledge across projects and harnesses. Use the wiki for meaning and scope; never duplicate its structure per agent.

## Authority Contract

Apply this precedence:

1. Alefita's current instruction.
2. Higher environment policies.
3. Applicable `AGENTS.md` and `CLAUDE.md`.
4. Wikifita directives.
5. Verified project memory.
6. Global profile and feedback.
7. Harness cache or automatic memory.
8. Model inferences.

Treat live project state as authority for operational facts that may have changed. Treat Alefita's explicit, current preferences as authority over prior inferences.

### Project identity and path routing

When Alefita names a repository or workspace path, that path is authoritative.
Before analysis, delegation, or any write, verify `pwd`,
`git rev-parse --show-toplevel`, branch, remotes, and the top-level inventory
against the designated path. Do not substitute the Codex process cwd, a
`Documents/Codex` worktree, a stale or alternate checkout, memory-derived path,
or a newly created repository. If the locations do not match, stop before
editing, copying, migrating, reconstructing, or creating files and report the
mismatch. In user-facing material, keep paths project-relative; abbreviate an
external path with `~/` only when indispensable.

## CLAUDE.md and AGENTS.md Compatibility

Use `CLAUDE.md` as the canonical instruction file. Maintain `AGENTS.md` as a relative symlink to the sibling `CLAUDE.md`:

```text
AGENTS.md -> CLAUDE.md
```

Apply the following rules:

- Discover instructions from the root relevant to the target directory.
- Treat a file closer to the target as a specialization of the more general one.
- Never maintain independent copies of `CLAUDE.md` and `AGENTS.md`.
- Always edit instructions in the canonical `CLAUDE.md`. Never edit, replace, move, rename, remove, or recreate `AGENTS.md` directly.
- When adding a new instruction scope, create the `CLAUDE.md` first and run `uv run scripts/wikifita_audit.py --fix` to materialize the `AGENTS.md`.
- After any creation, modification, movement, or removal of `CLAUDE.md`, run the audit with `--fix` and then without `--fix`; do not consider the work done until symlinks are intact.
- Never silently replace a real `AGENTS.md` or a divergent symlink; report the conflict.
- Validate that the symlink is relative, not broken, and resolves exactly to the sibling `CLAUDE.md`.
- Treat `CLAUDE.md` and `AGENTS.md` as reserved files, outside common OKF validation.
- Use normal audit execution for validation and `--fix` for safely creating missing symlinks.

## Branch Model

| Branch | Purpose | Deployed | Content |
|---|---|---|---|
| `main` | Public knowledge base + site content | Yes → wikifita.aleffita.dev | Technical docs, research, forensics, infrastructure |
| `personal` | Private notes, correspondence, sensitive references | No | Emails, salary negotiations, personal contacts, internal decisions |

**Rule:** Anything on `main` is potentially visible on the public site. Sensitive content goes on `personal`. Never merge `personal` to `main` without explicit review and sanitization.

### Publishing Flow
- `git push origin main` → triggers site rebuild at wikifita.aleffita.dev
- `personal` branch → never pushed to origin, or pushed to a private remote only

## Wikifita Model

Separate the roles:

- `index.md`: global index of wiki pages and domains.
- `log.md`: chronological log of significant changes.
- `memorias/MEMORY.md`: master index of all persistent memory.
- `memorias/user_alefita.md`: global profile; never move to project.
- `memorias/feedback/`: preferences, corrections, and global protocols.
- `memorias/projetos/<project>/`: project-specific context, decisions, and state.
- `pessoas/index.md`: complete, operational catalog of people and connections.
- `pessoas/conexoes/`: individual contact and relationship records.
- `diretivas/`: constitutional rules for work and artifacts.
- Other directories: knowledge organized by domain.

`MEMORY.md` is the master memory index. Indexes like `pessoas/index.md` are delegated sub-indexes: they must be reachable from `MEMORY.md` but preserve domain operational details.

## Editorial Workflow

The full editorial front is defined in `diretivas/editorial-workflow.md`. Summary of the mandatory phases:

### Phase 0: Pre-Conditions
- Confirm correct branch (`git branch --show-current`)
- Confirm the user-designated repository identity and live root path before reading or writing project material
- Confirm that wiki evolution is inside the authorized task scope
- Agents may discover and ingest relevant material autonomously within that scope
- Establish provenance before writing: public claims require authoritative online sources when available; local observations require inspectable project evidence; user-provided facts require explicit attribution
- Never fill a factual gap with generated content; preserve uncertainty or omit the claim when evidence is insufficient
- Consult `pessoas/index.md` before creating any person page
- Scan existing wiki for overlap

### Phase 1: Source Analysis
- Read source completely
- Identify conceptual atoms (discrete topics)
- Classify by type (reference/narrative/forensic/profile/post-mortem)
- Map cross-references
- **Never create a monolithic page** — if you can remove a topic without affecting the rest, it's a separate page

### Phase 2: Generation
- Sequential (< 5 pages) or parallel (subagents, > 5 pages)
- Per-page: OKF frontmatter, density appropriate to type, `[[name]]` wikilinks, Mermaid diagrams where topology matters
- One concept per page, dense with cross-references

### Phase 3: Integration
- Update `index.md` (domain section + tag table)
- Update `log.md` with semantic entry
- Update `memorias/MEMORY.md` and `pessoas/index.md` if applicable
- Cross-link with existing content in other domains

### Phase 4: Quality Gate
- `uv run scripts/wikifita_audit.py` → fix → audit again (no --fix) → PASS → commit
- Never commit until audit passes twice
- Never use `git commit --no-verify`

### Phase 5: Presentation (Artifacts)
- Apply visual identity from `diretivas/identidade-visual.md`
- Artifacts are derived from wiki content — wiki is source of truth
- Use the smallest useful visualization when a flow, topology, state transition, dataset, training run, tournament statistic or agentic system becomes materially easier to understand visually; optimize for neurodivergent accessibility and avoid decorative cognitive load
- Detect the current harness capabilities before choosing a presentation surface; never assume a tool exists because another harness provides it
- Treat an artifact as a user-reviewable output outside the active project and Git repository by default
- Put an artifact inside the project only when Alefita explicitly requests a project or repository file
- Prefer, in order: an in-conversation interactive surface when useful, a generated reviewable file outside the project, or a hosted site only when publication is explicitly requested
- Keep artifacts revisable and distinguish source material, review format, publication format and exported copies

## Content Taxonomy

| Type | OKF type | Density | Examples |
|---|---|---|---|
| Technical Reference | `reference` | High — tables, code, every line carries info | packet-protocol, native-modules |
| Architecture | `reference` | High — diagrams + prose | architecture, skia-rendering |
| Post-Mortem | `analysis` | Medium-high — metrics + narrative | code-quality, debugging-stories |
| Narrative | `narrative` | Medium — flowing prose | development-timeline, store-review-saga |
| People/Profile | `profile` | Low-medium — contextual | camdom-people, mikael-partner |
| Forensic | `analysis` | High — evidence-based, IOCs | clickfix-attack-chain, clickfix-bddr |
| Infrastructure | `reference` | High — config, stack | fitalabs-infra, litellm-gateway |
| Career/Identity | `profile` | Low-medium — positioning | camdom-career-profile, user_alefita |

Do not force uniform density across types. A profile page should not read like an API reference.

## Editorial Rules (Validated)

| # | Rule | Status |
|---|---|---|
| 1 | One concept per page — if you can remove a topic without affecting the rest, it's a separate page | VALIDATED |
| 2 | Dense with cross-references — the wiki is a graph, not a list | VALIDATED |
| 3 | OKF frontmatter on every page — type, title, description (EN), tags, timestamp | VALIDATED |
| 4 | Autonomous, provenance-bound maintenance — agents may create or update content inside an authorized scope, but never invent facts or fill evidentiary gaps | VALIDATED |
| 5 | Consult pessoas/index.md before creating person pages | VALIDATED |
| 6 | English for technical, Portuguese for personal | VALIDATED |
| 7 | Audit twice before committing — once with --fix, once without | VALIDATED |
| 8 | Check branch before writing — subagents don't inherit branch state | VALIDATED |
| 9 | `[[name]]` wikilink format only — path-based links break the auditor | VALIDATED |
| 10 | Each content type gets its own density | PROPOSED |
| 11 | Hub page per domain — every section has a clear overview | PROPOSED |
| 12 | Wiki is source of truth — artifacts are derived, never the reverse | PROPOSED |
| 13 | User-designated project path is authoritative — verify the live root before analysis or writes; never substitute a different checkout | VALIDATED |

## Wikilink Convention

```markdown
✅ [[camdom-architecture]]
✅ [[clickfix-handoff]]

❌ [[camdom/camdom-architecture]]
❌ [[wikifita/diretivas/...]]
```

## Artifact Visual Identity

When creating HTML artifacts, apply the Anthropic dark mode system from `diretivas/identidade-visual.md`:

- **Palette:** bg `#171717`, surface `#232329`, warm `#d4a574`, green `#66bfa2`, red `#e8706a`
- **Typography:** Source Serif 4 (headings), Inter (body), JetBrains Mono (code)
- **Tables:** uppercase headers, letter-spacing 0.4px, `--text3` color, hover `rgba(212,165,116,.04)`
- **Callouts:** `border-left: 3px solid var(--warm)`, bg `rgba(212,165,116,.06)`
- **Metadata header:** Project, Author, Engine, Version, Date, Status
- **Footer:** Project name + version + date
- **Location:** Session outputs by default; project folder only if user requests
- **Path privacy:** In user-facing or public material, use paths relative to the project. If an external path is indispensable, abbreviate it with `~/`; never expose absolute home paths or temporary directories
- **Provenance:** Retain model names when materially relevant and verified; do not present harness-specific agent names, surfaces or implementation details as project methodology
- **Never:** neon gradients, AI slop, decorative animations, glowing borders, generic dark mode

## Reading Flow

1. Read applicable instructions.
2. Start from `index.md` to locate the domain.
3. When memory is involved, start from `memorias/MEMORY.md`.
4. Consult only the files needed:
   - identity: `memorias/user_alefita.md`;
   - interaction: `memorias/feedback/`;
   - project: `memorias/projetos/<project>/`;
   - people: `pessoas/index.md` and the relevant connection;
   - artifacts: applicable directives.
5. Do not load the entire wiki.
6. Briefly note when a material conclusion came from Wikifita.
7. Mark temporal facts as possibly obsolete and verify against live state when needed.
8. Do not expose private or sensitive content unless the task requires it.

## Memory Flow

Create a memory at the smallest correct scope:

1. Record project-specific context in `memorias/projetos/<project>/`.
2. Promote to the global layer when durable and cross-project.
3. Keep profile only in `user_alefita.md`.
4. Keep global feedback in `feedback/`.
5. Archive or remove dated facts when they become obsolete.
6. Update `memorias/MEMORY.md` whenever memory topology or recoverability changes.

When staging candidates, use a single area without per-harness separation:

```text
memorias/inbox/
```

Organize candidates by memory meaning. Store origin as metadata, for example `source_harness`, `source_task`, `source_project`, `observed_at`, `confidence`, `sensitivity` and `status`. Do not create `codex/`, `claude/` or equivalent directories.

Consolidation should promote, merge, maintain in project, archive, or reject each candidate. Preserve provenance and convert relative time references to absolute dates.

## People and Connections

Treat `pessoas/index.md` as a complete, always-synchronized catalog:

- Create, edit, archive, or change a connection status and update the index in the same operation.
- Keep the individual page and index in the same commit.
- Verify coverage, links, status, and orphan records.
- Maintain a structural link to the catalog in `memorias/MEMORY.md`.
- Highlight individual contacts in `MEMORY.md` only when it improves contextual recovery; the catalog remains exhaustive.

## Writing Flow

Do not modify Wikifita merely because it was consulted. Write when Alefita requests registration or maintenance, when an authorized task explicitly includes wiki evolution, or when a persistent wiki update is a necessary in-scope result of that task. Agents may autonomously select sources, structure, cross-links and maintenance operations inside that scope; this autonomy never permits unsupported factual synthesis.

When writing:

1. Read applicable instructions.
2. Inspect neighboring pages and indexes before choosing location and format.
3. Classify each statement as sourced fact, local observation, inference, hypothesis, metaphor or unresolved claim; keep those categories explicit whenever confusion is possible.
4. Attach inspectable provenance to factual claims. Prefer primary or authoritative online sources for externally verifiable claims and live repository evidence for project facts.
5. Use OKF with `type`, `title`, `description`, `tags` and `timestamp`, except in reserved files or documented special schemas.
6. Use descriptive snake_case names and do not include dates in the name, unless there is a specific contract.
7. Update the domain index, `index.md`, `memorias/MEMORY.md` and `log.md` based on impact.
8. Treat each file individually; do not use regex for bulk operations.
9. Preserve external changes and do not mix them in the commit.
10. Audit before committing.
11. Make a semantic commit after a significant change when the request authorizes the full flow.

## Validation and Git

Use exclusively `uv` for Python:

```text
uv run scripts/wikifita_audit.py
```

Never invoke `python` or `python3` directly. Never use `git commit --no-verify`.

The audit should cover:

- OKF frontmatter;
- broken links;
- index coverage;
- obsolete references;
- `pessoas/index.md` synchronization;
- sub-index reachability from `memorias/MEMORY.md`;
- `AGENTS.md -> CLAUDE.md` symlink integrity.

Do not confuse current validation with planned validation: if a check is not yet implemented in the script, report the gap rather than declaring success.

## Limits

- Do not use Wikifita as indiscriminate ingestion.
- Do not invent, smooth over or silently complete missing evidence.
- Do not encode a harness-specific artifact API as a universal workflow; detect capabilities at runtime.
- Do not use anthropomorphic explanations for model or harness behavior. Report observable operations, evidence, uncertainty, and control points directly.
- Do not create a memory tree per harness.
- Do not turn agent automatic memory into canonical authority without consolidation.
- Do not store secrets.
- Do not duplicate critical rules only in memory; keep them also in instructions or appropriate documentation.
- Do not let indexes diverge from the content they represent.
