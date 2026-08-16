# `/goal` Bootstrap Prompt — Continue the Existing Autoresearch

Copy the text below into `/goal`.

---

Read `00_READ_FIRST.md`, the new `PROGRAM.md`, and every numbered research-context Markdown file in this compendium completely before making a substantive change.

This is a continuation of the existing Pokémon TCG autoresearch run, not a restart.

Recover the current Git state, current AR ledger, State Capsules, AR-015/AR-017 artifacts, trajectory collector, PPO/update path, tournament harness, and Stage 4 implementation. The previous run produced substantial code and approximately 18 commits; reuse it.

Where older reports or the previous `PROGRAM.md` conflict with this new compendium, this compendium is authoritative.

The research direction is explicit:

\[
\boxed{
\text{Stage 4}
\rightarrow
\text{RoPE-ND}
\rightarrow
\text{true recurrent self-play}
\rightarrow
\text{GRPO}
\rightarrow
\text{tournament + validation}
\rightarrow
\text{autoresearch iteration}.
}
\]

Normalize transcription correctly:

- GRPU-like forms mean GRPO.
- ROPEND means RoPE-ND.
- Do not introduce HOPE/Nested Learning or another unrelated architecture from transcript noise.

Do not spend the new run on historical BC ETL, Parquet, packed-data provenance, curriculum reconstruction, or backend polishing. Stage 4 already exists because behavioral cloning already happened. Historical BC data is at most a cheap validation diagnostic.

The new training data is self-play.

The previous `mirror_no_memory` collector is not the final self-play design. Implement true recurrent self-play with independent side-specific recurrent state that persists according to the actual Stage 4 inference semantics.

The previous PPO work is not GRPO. Implement the two algorithms defined in `03_SELFPLAY_GRPO_SPEC.md`:

1. trajectory-group GRPO;
2. sibling-fiber GRPO.

Train and tournament both. Compare them against each other and against the same useful opponent panel. Let autoresearch decide which branch to deepen. If they expose complementary signal, test a hybrid.

Implement RoPE-ND as specified in `04_ROPEND_SPEC.md`. Add it to the existing Stage 4 representation rather than replacing learned content encodings. Compare the best GRPO path with and without RoPE-ND.

Do not freeze heads, scratch, TBPTT, or other components by default. Do not redesign them gratuitously either. Change them when an experiment inside the self-play GRPO agenda justifies the change.

Optimize the entire self-play → update → tournament loop for maximum wall-clock research throughput on the dedicated M3 Pro with 24 GB unified memory. The objective is not resource conservation. Use memory aggressively, batch self-play, keep models resident, use in-memory rollout buffers, and remove unnecessary serialization/framework boundaries when measurement shows they cost time.

Track competitive win rate, per-opponent results, candidate-vs-current-best, side effects, validation accuracy when already cheap, entropy, KL, group variance, gradient norms, and throughput. The competition is dynamic, so candidate-vs-root alone is insufficient.

Follow the repository's existing GitFlow. Do not perform hours of work without commits. Commit every coherent implementation milestone and every meaningful experimental state, continue the AR numbering, and bind experiment ledger entries to commits.

Use sequential subagents:

worker → native HALT → reviewer → native HALT → integrate → commit → next objective.

No polling loops, no sleep schedulers, no artificial progress interrupts.

Review findings that invalidate mathematics/runtime are P0 and must be fixed. Cleanup/provenance issues must not become a new multi-hour prerequisite chain before self-play GRPO can run.

Your first concrete objective is:

**Implement true recurrent two-sided self-play on top of the existing trajectory collector, validate complete autoregressive behavior logprob and recurrent lane semantics with a small smoke, commit it, review it, then immediately implement trajectory-group GRPO.**

After the first trajectory-GRPO candidate exists, run a tournament.

Then implement sibling-fiber GRPO and run the same tournament surface.

Then compare, add RoPE-ND, and continue autoresearch from empirical evidence.

Do not respond with a long planning essay.

Start by reading the compendium, inspect the current code, create the next AR task, dispatch exactly one worker, and continue the execution loop.

Build, train, tournament, learn, commit, repeat.
