# AR-010 regeneration report

## Scope

AR-011 preserved the frozen Stage4 root and all prior autoresearch artifacts.
It closed the AR-010 provenance findings, regenerated a fresh candidate package
under AR-010, and did not run or submit a tournament.

The corrected probe used:

```text
checkpoint=experiments/autoresearch/root/stage4_root.pkl
checkpoint_sha256=b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b
agent_deck=agent/deck.csv
metadata_date=2026-08-12
seed=8018
games_per_mode=1
sample_modes=random,mirror_no_memory
```

The engine's documented replay limitation means the same seed is not a
guarantee of identical episode length. The final bounded run produced 177
samples: 85 random and 92 mirror-no-memory, with one terminal episode in each
mode.

## Provenance and input coverage

The candidate records the frozen root hash, manifest file hash, manifest
canonical-content hash, compressed bundle file hash, and canonical adjacent
artifact names. The manifest records separate normalized deck-content and
source-file hashes. Every one of the 177 ordered samples has action-mask,
detached-memory, and complete ordered model-input tensor digests.

| Artifact | SHA-256 |
| --- | --- |
| `candidate.pt` | `e6efe207d4b08dd458b40be14297b142ca2987b2238f97d807b6bf85320c7773` |
| `sample.manifest.json` file | `b5c273f75c8bee147a16fbde49b8ca5fed0c017fa656b5f29cf1d480db051256` |
| manifest canonical content | `915d7314f6356695f38b1b183c54efbb3bd8141ee28f89f981bc8722ccbdad16` |
| `trajectory_bundle.pt.gz` | `f2f6ca653d752d91fc17b61298c3dfa09aac1ad5741f0d4c6b71ba065aedbbbf` |
| `logs/trajectory.jsonl` | `6614e8d71da5b497b3fce70565ccc66e0961e703838d574f4723bd3305af08ec` |
| `logs/trajectory.manifest.json` | `d8fb4f308d13c78050b13dceaad6f6590b5ea43b7fb844c04d6c93b30966d1f1` |

Deck hashes:

```text
deck_content_sha256=606a775392ffe25e058b19c17801d58a4bf30f7cd8c62782388d3de7e7eb5283
deck_source_file_sha256=337186f9422f300e50225d6305570f008eb262ac46f519c62aa115df6dcc75d2
```

## PPO metrics

The one-epoch FP32 update used learning rate `1e-05`, gamma `1.0`, clip
epsilon `0.2`, and value coefficient `0.5`.

```text
loss=1.3862196207046509
policy_loss=0.00000025660304459051986
value_loss=2.7724387645721436
entropy=1.2229487895965576
gradient_norm=32.759674072265625
ratio_mean=1.0
ratio_min=0.9999985694885254
ratio_max=1.0000019073486328
root_reference_kl_mean=0.0001976492058020085
root_reference_parameter_l2=0.009192824462590189
root_reference_changed_parameters=1293409
root_reference_parameter_count=1302151
return_min=-1.0
return_max=-1.0
```

## Reviewer gates

- Absolute paths, `..` traversal, non-canonical artifact names, symlinks, and
  non-regular artifact files are rejected before artifact loading.
- Missing manifest/bundle, manifest-content tamper, bundle/manifest mismatch,
  root tamper, and file-hash tamper have adversarial tests.
- The old AR-009 candidate remains unchanged and is rejected by the corrected
  validator because it lacks the trajectory-bundle file hash.
- The regenerated candidate passes direct provenance validation and strict
  FP32 loading.
- With `PTCG_MODEL_PATH=experiments/autoresearch/AR-010/candidate.pt`, the real
  public agent's first `choose()` smoke test returned `[0]`; prospective
  planning remained disabled.
- With `PTCG_MODEL_PATH` unset, the default public path does not invoke the
  candidate provenance gate.

## Tournament status

The candidate is tournament-gate ready on provenance, strict-load, and
first-decision checks. No tournament was run or submitted by AR-011. Tournament
execution remains a subsequent operator action.
