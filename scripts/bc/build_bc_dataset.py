"""Build a behavioral-cloning dataset from PTCG episode replays -> a .npz of stacked
encoded-obs arrays + action labels (WINNING side only).

Mirrors inference-time encoding EXACTLY (rl.search_agent._net_greedy_select): per side a
GameTracker + AbilityTracker; each decision's recorded action (a list of option indices) is
expanded into per-pick rows `enc.encode(obs, picked=set(action[:k]), self_deck, tracker, ability_slots)`
with label=action[k], plus a SUBMIT row (label=SUBMIT_ACTION) when the side submitted before
maxCount. Deck step (select is None, action == 60 ids) resets the trackers and sets self_deck.

  python scripts/build_bc_dataset.py [ep_dir] [out.npz] [max_eps]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder, GameTracker, AbilityTracker, SUBMIT_ACTION
from rl.encoder.option_dedup import dup_legal_indices

EP_DIR = sys.argv[1] if len(sys.argv) > 1 else "_kaggle_scout/ep"
OUT = sys.argv[2] if len(sys.argv) > 2 else "_kaggle_scout/bc.npz"
MAX_EPS = int(sys.argv[3]) if len(sys.argv) > 3 else 0

ct = get_card_table()
enc = TokenEncoder(ct)

# Optional would_KO feature annotation (BC_WOULD_KO=1): run the 1-ply engine sim per attack option
# (rl.search_agent.annotate_would_ko) on each decision obs BEFORE encoding, so opt_attr's would_ko
# trio (rate / exp-prizes / P-win) is populated EXACTLY as in env collection + inference. Off by
# default -> the trio stays 0.0 (so the same builder produces the nowk arm, columns just zeroed).
WOULD_KO = os.environ.get("BC_WOULD_KO", "0") == "1"
WK_NVAR = int(os.environ.get("BC_WK_NVAR", "10"))
if WOULD_KO:
    from rl import search_agent as SA


def _dedup_group(sel, s, me, deck_list, action_mask, n):
    """Per-row '__group__' array (len = action dim): option index -> its FIRST-LEGAL canonical for
    single-pick stateless-card duplicates, identity everywhere else. Uses dup_legal_indices (the
    first LEGAL representative, NOT option_groups' first-overall which could be illegal) so the
    collapsed label always lands on a legal option. Multi-pick (maxCount>1) -> identity (dedup off).
    The trainer derives both the dedup action-mask (zero non-canonical legal dups) and the collapsed
    label (group[label]) from this, and an effect-equivalence-aware accuracy for BOTH A/B arms."""
    A = len(action_mask)
    arr = np.arange(A, dtype=np.int32)
    # bound by SUBMIT_ACTION (= option-slot capacity): real option indices live below it, and j must
    # never index past the action_mask (a decision can have more options than the encoder represents).
    legal = [j for j in range(min(n, SUBMIT_ACTION)) if float(action_mask[j]) >= 0.5]
    _, remap = dup_legal_indices(sel, s, me, deck_list, legal)
    for j, c in remap.items():
        arr[j] = c
    return arr


BOTH_SIDES = os.environ.get("BC_BOTH_SIDES", "1") == "1"   # clone BOTH players (default): Kaggle
# matchmaking pairs similar-rated agents, so the loser's moves are ~the winner's quality and
# winner-only throws away half the data for a near-nil quality filter. BC_BOTH_SIDES=0 = old behavior.


def rows_from_episode(ep, episode_id=None, ep_meta=None):
    """Yield (encoded_obs_dict, label, is_attack) for each cloned side's decisions.

    BOTH_SIDES by default: each side gets its OWN trackers fed only that side's
    decision obs -- train==test.  If ep_meta (list) is provided, episode metadata
    dicts are appended for D.3 sequential metadata sidecar.

    OBS<->ACTION OFF-BY-ONE: kaggle_environments records the action a player returns IN RESPONSE TO a
    select-obs on that player's NEXT entry, not the same one. So the label for a side's entry i
    (whose obs carries the select) is went[i+1]['action']. Same-entry pairing (the old bug) mislabels
    ~18-20% of decisions, which build_mask then legitimately masks -- it was a LABELING bug, never a
    tracker/encoder desync, and TRAINING is unaffected (the live env never replays recorded actions).
    The trackers are still fed each decision obs in order; the mask check below is now a self-validating
    tripwire (a correctly-paired label is ALWAYS legal, so if it fires the replay format changed)."""
    rewards = ep.get("rewards") or []
    if len(rewards) != 2 or rewards[0] is None or rewards[1] is None or rewards[0] == rewards[1]:
        return                                   # draw / malformed -> skip
    win = 0 if rewards[0] > rewards[1] else 1
    ep_id = episode_id if episode_id is not None else "unknown"
    for _side in ((0, 1) if BOTH_SIDES else (win,)):
        yield from _side_rows(ep, _side, ep_id, _side, ep_meta)


def _side_rows(ep, win, ep_id, side, ep_meta):
    went = [st[win] for st in (ep.get("steps") or []) if len(st) > win]   # this side's entries, in order
    tr, ab = GameTracker(), AbilityTracker()
    deck = None
    step = 0
    for i, ag in enumerate(went):
        obs = ag.get("observation") or {}
        sel = obs.get("select")
        action = ag.get("action")
        # DECK choice = a 60-id action (wherever it lands) -> set self_deck + reset the trackers.
        if isinstance(action, list) and len(action) == 60:
            deck = [int(c) for c in action]
            tr.reset(); ab.reset()
            continue
        if sel is None or deck is None:          # non-decision obs -> NOT fed to the tracker (matches inference)
            continue
        # advance the trackers on EVERY decision obs, in order (the logs are per-obs deltas)
        try:
            tr.update(obs)
            ab.note_turn((obs.get("current") or {}).get("turn"))
        except Exception:
            return                               # tracker error -> can't trust the rest of the episode
        # LABEL: the response to THIS select is recorded on the winner's NEXT entry (the off-by-one).
        label = went[i + 1].get("action") if i + 1 < len(went) else None
        opts = sel.get("option") or []
        n = len(opts)
        if not (isinstance(label, list) and label
                and all(isinstance(x, int) and 0 <= x < n for x in label)):
            continue                             # no valid pick recorded for this select (pass / boundary) -> no label
        try:
            maxc = sel.get("maxCount", 1)
            # ATTACK decision? = MAIN select (type 0) with >=1 attack option -> the subset where the
            # would_ko feature can bite. Flagged per row so the trainer can break out subset accuracy.
            is_atk = 1 if (sel.get("type") == 0 and any(o.get("attackId") is not None for o in opts)) else 0
            if WOULD_KO:                          # populate opt_attr's would_ko trio in-place (ONCE per decision)
                try:
                    SA.annotate_would_ko(obs, deck, enc, n_var=WK_NVAR)
                except Exception:
                    pass                          # bad/incomplete replay state -> leave trio at 0 (graceful)
            me = (obs.get("current") or {}).get("yourIndex")
            deck_list = sel.get("deck") if isinstance(sel.get("deck"), list) else None
            cur = obs.get("current") or {}
            rows = []
            new_ep = (step == 0)
            for k, idx in enumerate(label):
                e = enc.encode(obs, set(label[:k]), self_deck=deck, tracker=tr, ability_slots=ab.slots)
                if float(e["action_mask"][int(idx)]) < 0.5:
                    return                       # tripwire: a correctly-paired label is always legal -> replay format changed
                e["__group__"] = _dedup_group(sel, cur, me, deck_list, e["action_mask"], n)
                rows.append((e, int(idx), is_atk))
            if len(label) < maxc:
                e = enc.encode(obs, set(label), self_deck=deck, tracker=tr, ability_slots=ab.slots)
                if float(e["action_mask"][SUBMIT_ACTION]) < 0.5:
                    return
                e["__group__"] = _dedup_group(sel, cur, me, deck_list, e["action_mask"], n)
                rows.append((e, SUBMIT_ACTION, is_atk))
            for r in rows:
                yield r
                if ep_meta is not None:
                    ep_meta.append({
                        "episode_id": ep_id,
                        "side": int(side),
                        "step_id": step,
                        "new_episode": new_ep,
                    })
                    new_ep = False
            step += 1
            ab.record(sel, label)
        except Exception:
            return                               # malformed mid-episode -> drop the rest


def main():
    eps = sorted(glob.glob(os.path.join(EP_DIR, "*.json")))
    if MAX_EPS:
        eps = eps[:MAX_EPS]
    print(f"[bc] {len(eps)} episodes from {EP_DIR}", flush=True)
    rows, labels, attack, meta_list = [], [], [], []
    used = 0
    for i, f in enumerate(eps):
        try:
            ep = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        ep_id = os.path.splitext(os.path.basename(f))[0]
        got = 0
        for row, lab, ia in rows_from_episode(ep, episode_id=ep_id, ep_meta=meta_list):
            rows.append(row); labels.append(lab); attack.append(ia); got += 1
        used += 1 if got else 0
        if (i + 1) % 25 == 0:
            print(f"[bc]   {i + 1}/{len(eps)} eps -> {len(rows)} rows", flush=True)
    print(f"[bc] {len(rows)} rows from {used} winning sides", flush=True)
    if not rows:
        print("[bc] NO ROWS — check episode format"); return
    int_keys = set(enc.int_keys)
    keys = list(rows[0].keys())
    out = {}
    for k in keys:
        dt = np.int32 if (k in int_keys or k == "__group__") else np.float32
        out[k] = np.stack([r[k] for r in rows]).astype(dt)
    out["__labels__"] = np.array(labels, dtype=np.int64)
    out["__is_attack__"] = np.array(attack, dtype=np.int8)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    np.savez_compressed(OUT, **out)
    # D.3: emit episode metadata sidecar
    if meta_list:
        EP_META = np.dtype([
            ("episode_id", "U64"),
            ("side", "i4"),
            ("step_id", "i4"),
            ("new_episode", "bool"),
        ])
        meta_arr = np.array(meta_list, dtype=EP_META)
        meta_out = os.path.splitext(OUT)[0] + "_episode_meta.npy"
        np.save(meta_out, meta_arr, allow_pickle=False)
        print(f"[bc] episode_meta saved: {meta_out} ({len(meta_arr)} rows)", flush=True)
    import collections
    lc = collections.Counter(labels)
    print(f"[bc] saved {OUT}: {len(labels)} rows, {len(keys)} keys; "
          f"SUBMIT={lc.get(SUBMIT_ACTION,0)}, n_distinct_labels={len(lc)}, "
          f"attack_rows={int(np.sum(attack))} wouldko={'on' if WOULD_KO else 'off'}", flush=True)


if __name__ == "__main__":
    main()
