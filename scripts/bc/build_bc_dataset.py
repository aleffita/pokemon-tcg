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
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone

import numpy as np

from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder, GameTracker, AbilityTracker, SUBMIT_ACTION
from rl.encoder.option_dedup import dup_legal_indices
from rl.train_config import load_config

EP_DIR = sys.argv[1] if len(sys.argv) > 1 else "_kaggle_scout/ep"
OUT = sys.argv[2] if len(sys.argv) > 2 else "_kaggle_scout/bc.npz"
MAX_EPS = int(sys.argv[3]) if len(sys.argv) > 3 else 0

# Config defaults — overridden by configure() when called from build_bc_from_zips
WOULD_KO = False
WK_NVAR = 10
BOTH_SIDES = True
BUILD_SEED = 0
SELF_ALIASES = frozenset({"FitaLabs", "Alef Oliveira"})

ct = get_card_table()
enc = TokenEncoder(ct)

EPISODE_META_DTYPE = np.dtype([
    ("episode_id", "U64"),
    ("side", "i4"),
    ("step_id", "i4"),
    ("new_episode", "bool"),
    ("player_name", "U128"),
    ("opponent_name", "U128"),
    ("outcome", "i1"),
    ("is_self", "bool"),
    ("player_deck_hash", "U64"),
    ("opponent_deck_hash", "U64"),
    # D.3/CLAUDE.md sequential-metadata columns -- one entry per emitted row
    # (decision_id/substep let a loader group rows back into TBPTT chunks
    # without re-deriving the autoregressive multi-select structure).
    ("decision_id", "i4"),
    ("substep", "i1"),
    ("terminal", "bool"),
    ("reward", "f4"),
    # Replay-log-derived (no re-simulation) auxiliary RL targets -- see
    # _decision_prize_states / _compute_aux_targets below.
    ("aux_ko", "i1"),
    ("aux_prize_delta", "f4"),
    ("aux_terminal", "bool"),
    ("aux_return", "f4"),
    ("aux_valid", "i1"),
])

WOULD_KO_META_DTYPE = np.dtype([
    ("episode_id", "U64"),
    ("side", "i4"),
    ("step_id", "i4"),
    ("row_start", "i8"),
    ("row_count", "i4"),
    ("option_index", "i4"),
    ("requested_trials", "i4"),
    ("valid_trials", "i4"),
    ("failed_trials", "i4"),
    ("computed", "bool"),
    ("zero_result", "bool"),
    ("consistent_determinizations", "i4"),
    ("best_overlap_determinizations", "i4"),
    ("opponent_synthetic_cards", "i4"),
    ("self_synthetic_cards", "i4"),
    ("resolved_subselects", "i4"),
    ("ambiguous_subselects", "i4"),
    ("error_type", "U64"),
])


def configure(
    bc_would_ko=None,
    bc_wk_nvar=None,
    bc_both_sides=None,
    seed=None,
    self_aliases=None,
):
    """Update module-level config. Called by build_bc_from_zips after loading config."""
    global WOULD_KO, WK_NVAR, BOTH_SIDES, BUILD_SEED, SELF_ALIASES, SA
    if bc_would_ko is not None:
        WOULD_KO = bc_would_ko
    if bc_wk_nvar is not None:
        WK_NVAR = bc_wk_nvar
    if bc_both_sides is not None:
        BOTH_SIDES = bc_both_sides
    if seed is not None:
        BUILD_SEED = int(seed)
    if self_aliases is not None:
        SELF_ALIASES = frozenset(str(alias) for alias in self_aliases if str(alias))
    if WOULD_KO:
        from rl import search_agent as SA


def _decision_seed(episode_id, side, step):
    """Stable per-decision seed, independent of worker count and scheduling."""
    material = f"{BUILD_SEED}:{episode_id}:{side}:{step}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "little")


def _player_names(ep):
    """Return stable side-aligned player names from Kaggle replay metadata."""
    info = ep.get("info") or {}
    team_names = info.get("TeamNames")
    if isinstance(team_names, list) and len(team_names) >= 2:
        return str(team_names[0] or ""), str(team_names[1] or "")
    agents = info.get("Agents") or []
    names = []
    for side in range(2):
        agent = agents[side] if side < len(agents) and isinstance(agents[side], dict) else {}
        names.append(str(agent.get("Name") or agent.get("name") or ""))
    return names[0], names[1]


def _replay_decks(ep):
    """Extract each side's submitted 60-card deck from the replay when present."""
    decks = [None, None]
    for side in range(2):
        for step in ep.get("steps") or []:
            if len(step) <= side:
                continue
            action = step[side].get("action")
            if isinstance(action, list) and len(action) == 60:
                decks[side] = [int(card) for card in action]
                break
    return decks


def _deck_hash(deck):
    """Hash a deck multiset for provenance without feeding hidden cards to features."""
    if not deck:
        return ""
    payload = ",".join(str(card) for card in sorted(int(card) for card in deck))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def episode_meta_array(meta_list):
    """Encode row-aligned episode metadata without Python objects."""
    return np.array([
        (
            m["episode_id"],
            m["side"],
            m["step_id"],
            m["new_episode"],
            m["player_name"],
            m["opponent_name"],
            m["outcome"],
            m["is_self"],
            m["player_deck_hash"],
            m["opponent_deck_hash"],
            m.get("decision_id", m["step_id"]),
            m.get("substep", 0),
            m.get("terminal", False),
            m.get("reward", 0.0),
            m.get("aux_ko", 0),
            m.get("aux_prize_delta", 0.0),
            m.get("aux_terminal", False),
            m.get("aux_return", 0.0),
            m.get("aux_valid", 0),
        )
        for m in meta_list
    ], dtype=EPISODE_META_DTYPE)


def would_ko_meta_array(meta_list, would_ko_meta):
    """Encode compact per-attack-option audit records linked to dataset row ranges."""
    row_ranges = {}
    for row_index, meta in enumerate(meta_list):
        key = (str(meta["episode_id"]), int(meta["side"]), int(meta["step_id"]))
        if key not in row_ranges:
            row_ranges[key] = [row_index, 0]
        row_ranges[key][1] += 1

    records = []
    for decision in would_ko_meta:
        key = (
            str(decision["episode_id"]),
            int(decision["side"]),
            int(decision["step_id"]),
        )
        row_start, row_count = row_ranges.get(key, (-1, 0))
        audit = decision["audit"]
        for option_index, option in sorted(audit.get("options", {}).items()):
            records.append((
                key[0],
                key[1],
                key[2],
                row_start,
                row_count,
                int(option_index),
                int(option.get("requested_trials", 0)),
                int(option.get("valid_trials", 0)),
                int(option.get("failed_trials", 0)),
                bool(option.get("computed", False)),
                bool(option.get("zero_result", False)),
                int(audit.get("consistent_determinizations", 0)),
                int(audit.get("best_overlap_determinizations", 0)),
                int(audit.get("opponent_synthetic_cards", 0)),
                int(audit.get("self_synthetic_cards", 0)),
                int(audit.get("resolved_subselects", 0)),
                int(audit.get("ambiguous_subselects", 0)),
                str(audit.get("error_type", ""))[:64],
            ))
    return np.array(records, dtype=WOULD_KO_META_DTYPE)


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


# BOTH_SIDES default set above, overridden by configure()


def _decision_prize_states(went):
    """Cheap replay-log pre-pass (no tracker/encoder/re-simulation): walks
    ``went`` with the EXACT same decision-validity filter as the heavy loop
    below and records, per accepted decision, the turn number and each side's
    live prize COUNT straight from ``obs['current']['players']`` -- the same
    field the encoder itself reads. This lets aux targets be derived from
    what the replay already recorded, independent of whether the heavy loop
    later bails out on a tracker/encoder error partway through the episode.
    """
    states = []
    deck_seen = False
    for i, ag in enumerate(went):
        obs = ag.get("observation") or {}
        sel = obs.get("select")
        action = ag.get("action")
        if isinstance(action, list) and len(action) == 60:
            deck_seen = True
            continue
        if sel is None or not deck_seen:
            continue
        label = went[i + 1].get("action") if i + 1 < len(went) else None
        opts = sel.get("option") or []
        n = len(opts)
        if not (isinstance(label, list) and label
                and all(isinstance(x, int) and 0 <= x < n for x in label)):
            continue
        cur = obs.get("current") or {}
        turn = cur.get("turn")
        players = cur.get("players") or []
        me = cur.get("yourIndex")
        if turn is None or not isinstance(me, int) or not (0 <= me < 2) or len(players) < 2:
            states.append({"i": i, "valid": False})
            continue
        my_prize = len((players[me] or {}).get("prize") or [])
        opp_prize = len((players[1 - me] or {}).get("prize") or [])
        states.append({
            "i": i, "valid": True, "turn": int(turn),
            "my_prize": my_prize, "opp_prize": opp_prize,
        })
    return states


def _compute_aux_targets(states, outcome):
    """Per-decision aux RL targets, keyed by ``i`` (position in ``went``).

    Two DIFFERENT deltas are computed on purpose:

    - ``aux_prize_delta``/``aux_ko``: a lookahead over "the rest of the current
      turn" (consecutive valid states sharing the same turn number). This is
      spec-literal and intentionally REPEATS across every decision inside one
      turn -- it is a per-decision auxiliary prediction target, never summed.
    - ``reward``/``aux_return``: a TRUE consecutive-decision transition delta
      (this decision's prize state vs. the NEXT valid decision's, i.e. a
      telescoping series). Summing THIS one into a gamma=1.0 return never
      double-counts a single KO across the many decisions that share the turn
      it happened in -- reusing the turn-window value for the return would
      double- (or N-times-) count that KO once per decision in the turn.

    aux_terminal marks the last decision of the (already win/loss-only,
    draws are filtered by the caller) episode; the terminal decision's reward
    also carries the +1/-1 win/loss bonus. A decision whose prize/turn state
    could not be read gets every aux field zeroed with aux_valid=0 rather
    than a guessed value.
    """
    n = len(states)
    aux = {}
    for d in range(n):
        s = states[d]
        if not s["valid"]:
            aux[s["i"]] = {
                "aux_ko": 0, "aux_prize_delta": 0.0,
                "aux_terminal": 0, "aux_valid": 0,
                "reward": 0.0, "aux_return": 0.0,
            }
            continue
        end = d
        for d2 in range(d, n):
            s2 = states[d2]
            if not s2["valid"] or s2["turn"] != s["turn"]:
                break
            end = d2
        end_state = states[end]
        opp_delta = s["opp_prize"] - end_state["opp_prize"]   # opp prizes taken from here to end-of-turn (>=0)
        my_delta = s["my_prize"] - end_state["my_prize"]      # my prizes lost from here to end-of-turn (>=0)
        is_terminal = (d == n - 1)
        aux[s["i"]] = {
            "aux_ko": 1 if (opp_delta != 0 or my_delta != 0) else 0,
            "aux_prize_delta": float(opp_delta - my_delta),
            "aux_terminal": int(is_terminal),
            "aux_valid": 1,
        }

    valid_idx = [d for d in range(n) if states[d]["valid"]]
    reward_by_i = {}
    for pos, d in enumerate(valid_idx):
        s = states[d]
        is_terminal = (d == n - 1)
        terminal_bonus = 0.0
        if is_terminal:
            terminal_bonus = 1.0 if outcome > 0 else (-1.0 if outcome < 0 else 0.0)
        if pos + 1 < len(valid_idx):
            nxt = states[valid_idx[pos + 1]]
            opp_step = s["opp_prize"] - nxt["opp_prize"]
            my_step = s["my_prize"] - nxt["my_prize"]
        else:
            opp_step = my_step = 0
        reward_by_i[s["i"]] = float(opp_step - my_step) / 6.0 + terminal_bonus

    running = 0.0
    for d in reversed(valid_idx):
        s = states[d]
        entry = aux[s["i"]]
        r = reward_by_i[s["i"]]
        running += r
        entry["reward"] = r
        entry["aux_return"] = running
    return aux


def rows_from_episode(ep, episode_id=None, ep_meta=None, would_ko_meta=None):
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
    names = _player_names(ep)
    replay_decks = _replay_decks(ep)
    for _side in ((0, 1) if BOTH_SIDES else (win,)):
        outcome = 1 if rewards[_side] > rewards[1 - _side] else (
            -1 if rewards[_side] < rewards[1 - _side] else 0
        )
        yield from _side_rows(
            ep,
            _side,
            ep_id,
            _side,
            ep_meta,
            would_ko_meta,
            names[_side],
            names[1 - _side],
            outcome,
            _deck_hash(replay_decks[_side]),
            _deck_hash(replay_decks[1 - _side]),
        )


def _side_rows(
    ep,
    win,
    ep_id,
    side,
    ep_meta,
    would_ko_meta,
    player_name,
    opponent_name,
    outcome,
    player_deck_hash,
    opponent_deck_hash,
):
    went = [st[win] for st in (ep.get("steps") or []) if len(st) > win]   # this side's entries, in order
    # aux RL targets, derived straight from the replay log's prize counts (no re-simulation),
    # keyed by position `i` in `went` -- computed up front so aux_terminal/aux_return reflect
    # the TRUE final decision even if the heavy loop below bails out early on a tracker error.
    aux_by_i = _compute_aux_targets(_decision_prize_states(went), outcome)
    _EMPTY_AUX = {
        "aux_ko": 0, "aux_prize_delta": 0.0, "aux_terminal": 0,
        "aux_valid": 0, "reward": 0.0, "aux_return": 0.0,
    }
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
            wk_audit = None
            if WOULD_KO and is_atk:               # populate opt_attr's would_ko trio in-place (ONCE per decision)
                try:
                    _, wk_audit = SA.annotate_would_ko_with_audit(
                        obs,
                        deck,
                        enc,
                        n_var=WK_NVAR,
                        rng=random.Random(_decision_seed(ep_id, side, step)),
                    )
                except Exception as exc:
                    # Preserve a failed-computation record instead of conflating it with
                    # a valid all-zero would-KO result.
                    atk_indices = [
                        j for j, option in enumerate(opts)
                        if option.get("attackId") is not None
                    ]
                    wk_audit = {
                        "eligible_options": len(atk_indices),
                        "requested_trials": 0,
                        "valid_trials": 0,
                        "failed_trials": len(atk_indices),
                        "annotation_failures": 1,
                        "error_type": type(exc).__name__,
                        "options": {
                            j: {
                                "requested_trials": 0,
                                "valid_trials": 0,
                                "failed_trials": 1,
                                "computed": False,
                                "zero_result": False,
                            }
                            for j in atk_indices
                        },
                    }
            if would_ko_meta is not None and is_atk:
                if wk_audit is None:
                    # Feature disabled is explicit and distinct from a failed computation.
                    wk_audit = {
                        "eligible_options": sum(
                            option.get("attackId") is not None for option in opts
                        ),
                        "requested_trials": 0,
                        "valid_trials": 0,
                        "failed_trials": 0,
                        "disabled": 1,
                        "options": {},
                    }
                would_ko_meta.append({
                    "episode_id": ep_id,
                    "side": int(side),
                    "step_id": step,
                    "audit": wk_audit,
                })
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
            aux = aux_by_i.get(i, _EMPTY_AUX)
            for row_idx, r in enumerate(rows):
                yield r
                if ep_meta is not None:
                    ep_meta.append({
                        "episode_id": ep_id,
                        "side": int(side),
                        "step_id": i,
                        "new_episode": new_ep,
                        "player_name": player_name,
                        "opponent_name": opponent_name,
                        "outcome": int(outcome),
                        "is_self": player_name in SELF_ALIASES,
                        "player_deck_hash": player_deck_hash,
                        "opponent_deck_hash": opponent_deck_hash,
                        # sequential metadata (CLAUDE.md "Sequential metadata and persistent scratch")
                        "decision_id": step,
                        "substep": row_idx,
                        "terminal": bool(aux["aux_terminal"]),
                        "reward": float(aux["reward"]),
                        # replay-log-derived aux RL targets
                        "aux_ko": int(aux["aux_ko"]),
                        "aux_prize_delta": float(aux["aux_prize_delta"]),
                        "aux_terminal": bool(aux["aux_terminal"]),
                        "aux_return": float(aux["aux_return"]),
                        "aux_valid": int(aux["aux_valid"]),
                    })
                    new_ep = False
            step += 1
            ab.record(sel, label)
        except Exception:
            return                               # malformed mid-episode -> drop the rest


def main():
    cfg = load_config()
    configure(
        bc_would_ko=cfg.bc_would_ko,
        bc_wk_nvar=cfg.bc_wk_nvar,
        bc_both_sides=cfg.bc_both_sides,
        seed=cfg.seed,
        self_aliases=cfg.bc_self_aliases,
    )
    eps = sorted(glob.glob(os.path.join(EP_DIR, "*.json")))
    if MAX_EPS:
        eps = eps[:MAX_EPS]
    print(f"[bc] {len(eps)} episodes from {EP_DIR}", flush=True)
    rows, labels, attack, meta_list, wk_meta_list = [], [], [], [], []
    used = 0
    for i, f in enumerate(eps):
        try:
            ep = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        ep_id = os.path.splitext(os.path.basename(f))[0]
        got = 0
        for row, lab, ia in rows_from_episode(
            ep,
            episode_id=ep_id,
            ep_meta=meta_list,
            would_ko_meta=wk_meta_list,
        ):
            rows.append(row); labels.append(lab); attack.append(ia); got += 1
        used += 1 if got else 0
        if (i + 1) % 25 == 0:
            print(f"[bc]   {i + 1}/{len(eps)} eps -> {len(rows)} rows", flush=True)
    print(f"[bc] {len(rows)} rows from {used} winning sides", flush=True)
    if not rows:
        print("[bc] NO ROWS — check episode format"); return
    wk_stats = {
        "eligible_decisions": len(wk_meta_list),
        "eligible_options": sum(
            int(item["audit"].get("eligible_options", 0)) for item in wk_meta_list
        ),
        "computed_options": sum(
            int(bool(option.get("computed", False)))
            for item in wk_meta_list
            for option in item["audit"].get("options", {}).values()
        ),
        "zero_result_options": sum(
            int(bool(option.get("zero_result", False)))
            for item in wk_meta_list
            for option in item["audit"].get("options", {}).values()
        ),
        "valid_trials": sum(
            int(item["audit"].get("valid_trials", 0)) for item in wk_meta_list
        ),
        "failed_trials": sum(
            int(item["audit"].get("failed_trials", 0)) for item in wk_meta_list
        ),
    }
    if WOULD_KO and (
        wk_stats["eligible_options"] <= 0
        or wk_stats["computed_options"] <= 0
        or wk_stats["valid_trials"] <= 0
    ):
        raise RuntimeError(
            "bc_would_ko=true but the dataset did not produce a successful would-KO computation"
        )
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
        meta_arr = episode_meta_array(meta_list)
        meta_out = os.path.splitext(OUT)[0] + "_episode_meta.npy"
        np.save(meta_out, meta_arr, allow_pickle=False)
        print(f"[bc] episode_meta saved: {meta_out} ({len(meta_arr)} rows)", flush=True)
        wk_meta_arr = would_ko_meta_array(meta_list, wk_meta_list)
        wk_meta_out = os.path.splitext(OUT)[0] + "_would_ko_meta.npy"
        np.save(wk_meta_out, wk_meta_arr, allow_pickle=False)
        print(
            f"[bc] would_ko_meta saved: {wk_meta_out} ({len(wk_meta_arr)} attack options)",
            flush=True,
        )
    source_signature = hashlib.sha256()
    total_source_bytes = 0
    for path in eps:
        stat = os.stat(path)
        total_source_bytes += stat.st_size
        source_signature.update(
            f"{os.path.basename(path)}:{stat.st_size}".encode("utf-8")
        )
    manifest = {
        "manifest_version": 1,
        "builder": "scripts/bc/build_bc_dataset.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "episode_dir": EP_DIR,
            "episode_files": len(eps),
            "total_bytes": total_source_bytes,
            "filename_size_sha256": source_signature.hexdigest(),
        },
        "config": {
            "bc_would_ko": WOULD_KO,
            "bc_wk_nvar": WK_NVAR,
            "bc_both_sides": BOTH_SIDES,
            "seed": BUILD_SEED,
            "self_aliases": sorted(SELF_ALIASES),
        },
        "output": {"path": OUT, "rows": len(labels)},
        "would_ko": {
            "status": "computed" if WOULD_KO else "disabled",
            "heuristic_version": 2,
            "counters": wk_stats,
        },
        "hidden_replay_data": {
            "deck_storage": (
                "episode sidecar:player_deck_hash,opponent_deck_hash"
            ),
            "used_by_would_ko": False,
        },
    }
    with open(f"{OUT}.manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    import collections
    lc = collections.Counter(labels)
    print(f"[bc] saved {OUT}: {len(labels)} rows, {len(keys)} keys; "
          f"SUBMIT={lc.get(SUBMIT_ACTION,0)}, n_distinct_labels={len(lc)}, "
          f"attack_rows={int(np.sum(attack))} "
          f"would_ko_status={'computed' if WOULD_KO else 'disabled'} "
          f"computed_options={wk_stats['computed_options']} "
          f"zero_result_options={wk_stats['zero_result_options']} "
          f"failed_trials={wk_stats['failed_trials']}", flush=True)


if __name__ == "__main__":
    main()
