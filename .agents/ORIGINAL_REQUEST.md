# Original User Request

## Initial Request — 2026-08-14T11:08:52-03:00

You are the Project Orchestrator (teamwork_preview_orchestrator).

Your working directory is: `/Users/alefita/workdir/pokemon-tcg/.agents/orchestrator_1/`
The project workspace root is: `/Users/alefita/workdir/pokemon-tcg`
The user's original request is recorded at: `/Users/alefita/workdir/pokemon-tcg/.agents/ORIGINAL_REQUEST.md`

## Mission & Requirements
Execute and orchestrate the full research, neural expansion, dataset compilation, mathematical monograph, and cross-project knowledge base synchronization:

1. **R1. Neural Architecture & 4D RoPEND MoE Expansion**:
   - Implement 4D Rotary Positional Embedding (RoPEND) operators ($c_1$: Step, $c_2$: Meta-Epoch, $c_3$: Urgency Clock, $c_4$: Inferred Elo) in MLX and PyTorch.
   - Expand the validated Stage 4 base model into the Mixture-of-Experts (MoE) topology for the August 16–31 Locked Meta phase.
   - Ensure strict FP32 precision contract across `rl/policy_infer_torch.py` and training pipeline.

2. **R2. Elite Pool Dataset Re-compilation & Oracles**:
   - Compile the clean Elite Match Dataset (Elo >= 1100, ~100k matches) from local replay archives (`data/bc_replay_zip/`).
   - Include corrected auxiliary targets (`aux_ko`, `aux_prize_delta`, `aux_terminal`, `aux_return`), C++ `bc_would_ko` damage annotations, and pre-game vehicle draft sequences.

3. **R3. PageRank & Abelian Graph Invariance Monograph**:
   - Author a dedicated technical monograph and Wikifita reference analyzing the mathematical isomorphism between PageRank Markov chain stationarity (dangling mass redistribution) and Bradley-Terry Softmax Abelian Group Elo calibration.

4. **R4. Canonical Wikifita Cross-Project Synchronization**:
   - Integrate cross-project memory, editorial guidelines, and mathematical proofs into canonical Wikifita pages (`~/Claude/wikifita/kaggle/` and `~/Claude/wikifita/co-scientist/`).
   - Execute double validation via `uv run scripts/wikifita_audit.py` (with `--fix` and without `--fix`).

## Strict Acceptance Criteria
- [ ] PyTorch inference engine (`rl/policy_infer_torch.py`) passes 100% strict FP32 checksum and static feature array SHA256 validation.
- [ ] Tournament benchmark (`scripts/tournament.py`) executes 500 matches against `first_sub_kaggle_2707` without process hangs, NaN logits, or memory leaks.
- [ ] The agent achieves > 40% overall win rate against `first_sub` on the Yan (#633) archetype.
- [ ] SQLite database (`model/results.db`) maintains 100.0% physical parity against disk JSON archives with 0 unhandled foreign key errors.
- [ ] `uv run scripts/wikifita_audit.py` passes twice in `~/Claude/wikifita/` with 0 broken `[[wikilinks]]` or orphaned records.
- [ ] Master RFC (`docs/technical_handoff_rfc.md`) and Metanoia suite (`docs/metanoia/01..06`) remain fully indexed and synchronized.
