"""Shared decision-time search primitives (independent of the net).

Observation-conditioned PIMC determinization (`_determinize`), the branchable-node
test (`_branchable`), and the MCTS node (`_Node`) -- imported by the live v2 MCTS in
search_agent.py. The v1-net MCTS that used to live here (search_select / mcts_* /
the obs_to_tensors-based value + priors) was removed along with the v1 policy; only
these net-agnostic primitives remain.
"""

from __future__ import annotations

from collections import Counter

from rl.deck.decks import DECKS
from rl.encoder.encoding import MAX_OPTIONS


def _load_sample_deck():
    """The engine 'sample' deck (agent/deck.csv) -- part of the meta/training pool but
    not in DECKS. Bundle-safe: alongside this module in a submission, ../agent in repo."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "deck.csv"), os.path.join(here, "..", "agent", "deck.csv")):
        try:
            if os.path.exists(p):
                return [int(x) for x in open(p) if x.strip()]
        except Exception:
            pass
    return None


# Deck hypotheses for opponent inference: the known meta = official archetypes + the
# engine sample deck (which is also the multi-deck training pool). Covers every deck an
# opponent might play, so inference isn't forced into a same-archetype fallback.
_CANDIDATES = [list(d) for d in DECKS.values()]
_sample_deck = _load_sample_deck()
if _sample_deck:
    _CANDIDATES.append(_sample_deck)
try:    # generated archetypes -> match the --decks all+gen training pool (absent in the submission bundle, which stays at the 5 meta decks)
    from rl.deck.decks_generated import GENERATED
    _CANDIDATES.extend(list(d) for d in GENERATED.values())
except Exception:
    pass


def _cid(c):
    return c.get("id") if isinstance(c, dict) else c


def _observed(pl, stadium, owner, with_hand):
    """Counter of card ids of player `owner` that are publicly visible (or, for our
    own side, also our hand). These are hard constraints: definitely in their 60."""
    cnt = Counter()
    for grp in ("active", "bench"):
        for pk in (pl.get(grp) or []):
            if not pk:
                continue
            cnt[pk["id"]] += 1
            for key in ("preEvolution", "energyCards", "tools"):
                for c in (pk.get(key) or []):
                    cnt[_cid(c)] += 1
    for c in (pl.get("discard") or []):
        cnt[_cid(c)] += 1
    if with_hand:
        for c in (pl.get("hand") or []):
            cnt[_cid(c)] += 1
    for c in (stadium or []):
        if isinstance(c, dict) and c.get("playerIndex") == owner:
            cnt[c["id"]] += 1
    return cnt


def _fit(cards, n, full, rng, audit=None, audit_key=None):
    """Trim/pad a card list to exactly n and report synthetic fills when requested."""
    cards = list(cards)
    rng.shuffle(cards)
    if len(cards) > n:
        return cards[:n]
    missing = max(0, n - len(cards))
    if audit is not None and audit_key is not None:
        audit[audit_key] = audit.get(audit_key, 0) + missing
    while len(cards) < n and full:
        cards.append(rng.choice(full))
    return cards


def _basic_pokemon(ids, enc):
    """A basic Pokemon id from `ids` (for a face-down opp active), else None."""
    for cid in ids:
        f = enc.cards.features(cid)
        if f is not None and f[0] > 0.5 and f[3] > 0.5:   # category=pokemon, stage=Basic
            return int(cid)
    return None


def _determinize(obs, deck, rng, enc, audit=None):
    """Observation-conditioned PIMC. The opponent's visible cards (discard, board,
    attachments, their stadium) are hard constraints; we infer which known decklist
    is consistent with them and fill the hidden zones from `inferred_deck - observed`.
    Our own hidden cards (deck/prizes) come from `our_deck - seen - hand`."""
    s = obs["current"]; me = s["yourIndex"]
    mp, op = s["players"][me], s["players"][1 - me]
    stadium = s.get("stadium")

    # --- opponent: infer archetype, then deal the unseen remainder ---
    seen = _observed(op, stadium, 1 - me, with_hand=False)
    cands = [Counter(d) for d in (_CANDIDATES + [list(deck)])]
    consistent = [c for c in cands if all(seen[k] <= c[k] for k in seen)]
    if not consistent:                       # off-meta opponent: best-overlap match
        consistent = [max(cands, key=lambda c: sum(min(seen[k], c[k]) for k in seen))]
        if audit is not None:
            audit["best_overlap_determinizations"] = audit.get("best_overlap_determinizations", 0) + 1
    elif audit is not None:
        audit["consistent_determinizations"] = audit.get("consistent_determinizations", 0) + 1
        audit["consistent_candidates_total"] = (
            audit.get("consistent_candidates_total", 0) + len(consistent)
        )
    D = rng.choice(consistent)               # ensemble across determinizations
    full_opp = list(D.elements())
    rem = []
    for cid, ct in D.items():
        rem += [cid] * max(0, ct - seen[cid])
    rng.shuffle(rem)

    dC = op["deckCount"]; pC = len(op.get("prize") or []); hC = op.get("handCount", 0)
    face = bool(op.get("active") and op["active"][0] is None)
    rem = _fit(
        rem,
        dC + pC + hC + (1 if face else 0),
        full_opp,
        rng,
        audit=audit,
        audit_key="opponent_synthetic_cards",
    )
    i = 0
    opp_deck = rem[i:i + dC]; i += dC
    opp_prize = rem[i:i + pC]; i += pC
    opp_hand = rem[i:i + hC]; i += hC
    opp_active = []
    if face:
        opp_active = [_basic_pokemon(rem[i:] + full_opp, enc) or rem[i]]

    # --- our side: deck/prizes are what's left of our deck after seen + hand ---
    seen_me = _observed(mp, stadium, me, with_hand=True)
    Dme = Counter(deck)
    rem_me = []
    for cid, ct in Dme.items():
        rem_me += [cid] * max(0, ct - seen_me[cid])
    rng.shuffle(rem_me)
    mdC = mp["deckCount"]; mpC = len(mp.get("prize") or [])
    rem_me = _fit(
        rem_me,
        mdC + mpC,
        list(deck),
        rng,
        audit=audit,
        audit_key="self_synthetic_cards",
    )

    return dict(
        your_deck=rem_me[:mdC],
        your_prize=rem_me[mdC:mdC + mpC],
        opponent_deck=opp_deck,
        opponent_prize=opp_prize,
        opponent_hand=opp_hand,
        opponent_active=opp_active,
    )


def _branchable(obs_dict):
    """A node we branch on: single-pick with >=2 real options, not terminal."""
    s = obs_dict.get("select")
    return (s is not None and s.get("maxCount", 1) == 1
            and 2 <= len(s["option"]) <= MAX_OPTIONS   # >MAX_OPTIONS: not encodable, net can't score
            and obs_dict["current"]["result"] < 0)


class _Node:
    __slots__ = ("sid", "obs", "me_turn", "term", "tv", "P", "N", "W", "kids", "exp")

    def __init__(self, sid, obs, me):
        self.sid = sid; self.obs = obs
        cur = obs["current"]
        self.term = cur["result"] >= 0
        self.tv = (0.0 if cur["result"] == 2 else (1.0 if cur["result"] == me else -1.0)) if self.term else 0.0
        self.me_turn = cur["yourIndex"] == me
        self.P = None; self.exp = False
        n = len(obs["select"]["option"]) if obs.get("select") else 0
        self.N = np.zeros(n); self.W = np.zeros(n); self.kids = {}


# ===========================================================================
# Decision-time PUCT MCTS + would_ko (merged from the former search_agent.py)
# ===========================================================================
import dataclasses
import random

import numpy as np
import torch

from rl.encoder.encoding import SUBMIT_ACTION
from rl.encoder.encoding import N_ACTIONS

try:    # per-attack is_variable flag for variable-aware would_KO sampling (bundle-safe)
    from rl.encoder.attack_data import ATTACKS as _WK_ATTACKS
except Exception:
    _WK_ATTACKS = {}

WK_NDET_VAR = 10    # determinizations for a VARIABLE-damage attack (coin/conditional) -> KO probability +
                    # expected-prizes + win estimates; fixed-damage attacks currently use one sampled hidden state.


def _tens(enc, enc_obs):
    return {k: torch.as_tensor(np.asarray(v)[None],
                               dtype=(torch.long if k in enc.int_keys else torch.float32))
            for k, v in enc_obs.items()}


@torch.no_grad()
def _net_greedy_select(obs, net, enc, deck, tracker, ability):
    """Full engine selection from the v2 net (buffered single-pick), no search."""
    sel = obs["select"]; picked: list[int] = []
    for _ in range(sel.get("maxCount", 1) + 1):
        o = _tens(enc, enc.encode(obs, set(picked), self_deck=deck, tracker=tracker, ability_slots=ability))
        a = int(net.logits_value(o)[0][0].argmax())
        if a == SUBMIT_ACTION:
            break
        picked.append(a)
        if len(picked) >= sel.get("maxCount", 1):
            break
    return sorted(set(picked))


@torch.no_grad()
def _value(net, enc, obs_dict, me, deck, tracker, ability) -> float:
    cur = obs_dict["current"]
    if cur["result"] >= 0:                                   # terminal
        return 0.0 if cur["result"] == 2 else (1.0 if cur["result"] == me else -1.0)
    o = _tens(enc, enc.encode(obs_dict, set(), self_deck=deck, tracker=tracker, ability_slots=ability))
    v = float(net.get_value(o)[0])
    return v if cur["yourIndex"] == me else -v               # net value is for the acting player


@torch.no_grad()
def _priors_value(net, enc, obs_dict, me, deck, tracker, ability):
    o = _tens(enc, enc.encode(obs_dict, set(), self_deck=deck, tracker=tracker, ability_slots=ability))
    logits, val = net.logits_value(o)
    n = len(obs_dict["select"]["option"])
    p = torch.softmax(logits[0, :n], -1).cpu().numpy() if n else np.zeros(0)
    v = float(val[0]); cur = obs_dict["current"]
    return p, (v if cur["yourIndex"] == me else -v)


def _advance(api, sid, obs, net, enc, deck, tracker, ability):
    """Step through forced/multi-pick selects (net-greedy) to the next branchable node."""
    while obs["current"]["result"] < 0 and not _branchable(obs):
        st = api.search_step(sid, _net_greedy_select(obs, net, enc, deck, tracker, ability))
        sid = st.searchId; obs = dataclasses.asdict(st.observation)
    return sid, obs


def mcts_visits(obs, net, enc, deck, tracker=None, ability=None,
                n_sims=160, n_det=2, c_puct=1.5, rng=None):
    """PUCT MCTS over the v2 net; returns (root visit counts over options, ok)."""
    rng = rng or random.Random()
    sel = obs.get("select")
    if sel is None or not _branchable(obs) or sel.get("type") != 0:
        return None, False
    from cg import api
    me = obs["current"]["yourIndex"]
    n_opt = len(sel["option"])
    agg = np.zeros(n_opt)

    def simulate(node):
        if node.term:
            return node.tv
        if not node.exp:
            node.P, v = _priors_value(net, enc, node.obs, me, deck, tracker, ability)
            node.exp = True
            return v
        N, W, P = node.N, node.W, node.P
        sqrtsum = float(np.sqrt(N.sum() + 1e-8))
        Q = np.where(N > 0, W / np.maximum(N, 1), 0.0)
        score = (Q if node.me_turn else -Q) + c_puct * P * sqrtsum / (1.0 + N)
        a = int(score.argmax())
        if a not in node.kids:
            st = api.search_step(node.sid, [a])
            csid, cobs = _advance(api, st.searchId, dataclasses.asdict(st.observation),
                                  net, enc, deck, tracker, ability)
            node.kids[a] = _Node(csid, cobs, me)
        v = simulate(node.kids[a])
        node.N[a] += 1; node.W[a] += v
        return v

    for _ in range(n_det):
        try:
            root_ss = api.search_begin(api.to_observation_class(obs), **_determinize(obs, deck, rng, enc))
        except Exception:
            continue
        root = _Node(root_ss.searchId, dataclasses.asdict(root_ss.observation), me)
        if len(root.N) != n_opt:                              # alignment guard
            try: api.search_end()
            except Exception: pass
            return None, False
        for _ in range(n_sims):
            simulate(root)
        agg += root.N
        try: api.search_release(root_ss.searchId)
        except Exception: pass
    try: api.search_end()
    except Exception: pass
    if agg.sum() == 0:
        return None, False
    return agg, True


def _my_prizes(obs, me) -> int:
    pls = (obs.get("current") or {}).get("players") or []
    return len(pls[me].get("prize") or []) if 0 <= me < len(pls) else 6


def _advance_resolve(api, sid, obs, me, rng=None, audit=None, max_steps=64):
    """Resolve post-attack sub-selects with seeded sampling and bounded execution.

    The old implementation always selected the first ``maxCount`` options. That made
    option ordering an undocumented heuristic. We now choose exactly the required
    ``minCount`` options, sample them reproducibly when alternatives exist, and expose
    every such resolution through audit counters.
    """
    rng = rng or random.Random()
    steps = 0
    while obs["current"]["result"] < 0 and obs["current"]["yourIndex"] == me:
        steps += 1
        if steps > max_steps:
            if audit is not None:
                audit["resolution_limit_failures"] = audit.get("resolution_limit_failures", 0) + 1
            raise RuntimeError(f"would_ko post-attack resolution exceeded {max_steps} steps")
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        min_count = int(sel.get("minCount", 1) or 0)
        max_count = int(sel.get("maxCount", 1) or 0)
        if min_count < 0 or max_count < min_count or max_count > len(opts):
            if audit is not None:
                audit["invalid_subselect_failures"] = audit.get("invalid_subselect_failures", 0) + 1
            raise ValueError(
                f"invalid post-attack select bounds min={min_count} max={max_count} options={len(opts)}"
            )
        pick = sorted(rng.sample(range(len(opts)), min_count)) if min_count else []
        if audit is not None:
            audit["resolved_subselects"] = audit.get("resolved_subselects", 0) + 1
            audit["resolved_subselect_choices"] = (
                audit.get("resolved_subselect_choices", 0) + len(pick)
            )
            if len(opts) > min_count:
                audit["ambiguous_subselects"] = audit.get("ambiguous_subselects", 0) + 1
        st = api.search_step(sid, pick)
        sid = st.searchId; obs = dataclasses.asdict(st.observation)
    return sid, obs


def would_ko_flags_with_audit(
    obs,
    deck,
    enc,
    n_var=WK_NDET_VAR,
    rng=None,
    early_stop=False,
) -> tuple[dict, dict]:
    """Engine-accurate would-KO per ATTACK option: simulate the attack 1 ply on the SDK sim and
    report the KO RATE (we take a prize / win). Net-free + minimal -> usable as a TRAINING FEATURE
    per attack-option (abilities/stadium/weakness/variable all resolved by the real engine).
    VARIABLE-aware: a fixed-damage attack currently uses one sampled hidden-state rollout;
    a VARIABLE (coin/conditional) attack is sampled `n_var` times -> KO probability (the engine
    re-rolls each determinization; manual_coin=False auto-flips off the persistent agent_ptr RNG).
    Returns {option_index: (ko_rate, exp_prizes_taken, win_rate)}; {} if no attack options / not a
    MAIN select. exp_prizes_taken (0..6) grades the prize VALUE of the KO (ex=2, mega-ex=3); win_rate
    = P(this move ENDS the game in my favour: I take the last prize OR the opp can't promote)."""
    sel = obs.get("select")
    audit = {
        "eligible_options": 0,
        "requested_trials": 0,
        "valid_trials": 0,
        "failed_trials": 0,
        "options": {},
    }
    if sel is None or sel.get("type") != 0:
        return {}, audit
    opts = sel.get("option") or []
    atk = [i for i, o in enumerate(opts) if o.get("attackId") is not None]
    if not atk:
        return {}, audit
    from cg import api
    me = obs["current"]["yourIndex"]
    p0 = _my_prizes(obs, me)
    rng = rng or random.Random()
    out = {}
    audit["eligible_options"] = len(atk)
    for a in atk:
        av = _WK_ATTACKS.get(opts[a].get("attackId"))
        ndet = max(1, n_var) if (av and av[1]) else 1     # variable -> sample rate; fixed -> one rollout
        kos = wins = trials = 0
        failures = 0
        prize_sum = 0.0
        seen = set()                                       # distinct per-sim (ko,prizes,won) -> early-stop when unanimous
        audit["requested_trials"] += ndet
        for _ in range(ndet):
            try:
                det = _determinize(obs, deck, rng, enc, audit=audit)
                ss = api.search_begin(api.to_observation_class(obs), **det)
            except Exception:
                failures += 1
                audit["search_begin_failures"] = audit.get("search_begin_failures", 0) + 1
                continue
            try:
                st = api.search_step(ss.searchId, [a])
                _, o2 = _advance_resolve(
                    api,
                    st.searchId,
                    dataclasses.asdict(st.observation),
                    me,
                    rng=rng,
                    audit=audit,
                )
                trials += 1
                cur = o2["current"]
                won = (cur["result"] == me)                # this move ENDS the game in my favour
                took = max(0, p0 - _my_prizes(o2, me))     # prizes I gain this ply (0/1/2/3; ex=2, mega=3)
                ko_bit = 1 if (won or (cur["result"] < 0 and took > 0)) else 0
                kos += ko_bit
                prize_sum += took
                wins += int(won)
                if early_stop: seen.add((ko_bit, took, int(won)))
            except Exception:
                failures += 1
                audit["simulation_failures"] = audit.get("simulation_failures", 0) + 1
            finally:
                try: api.search_release(ss.searchId)
                except Exception:
                    audit["search_release_failures"] = audit.get("search_release_failures", 0) + 1
            # EARLY-STOP: a determinization-invariant attack (all sims identical) is settled after 3
            # confirmations -> skip the remaining n_var sims (97% of variable attacks are deterministic;
            # the ~3% genuine coin-flips disagree early and run the full n_var, so the rate is unbiased there).
            if early_stop and ndet > 1 and trials >= 3 and len(seen) == 1:
                break
        if trials:
            out[a] = (kos / trials, prize_sum / trials, wins / trials)
        audit["valid_trials"] += trials
        audit["failed_trials"] += failures
        audit["options"][a] = {
            "requested_trials": ndet,
            "valid_trials": trials,
            "failed_trials": failures,
            "computed": bool(trials),
            "zero_result": bool(trials and kos == 0 and prize_sum == 0 and wins == 0),
        }
    try:
        api.search_end()
    except Exception:
        audit["search_end_failures"] = audit.get("search_end_failures", 0) + 1
    return out, audit


def would_ko_flags(
    obs,
    deck,
    enc,
    n_var=WK_NDET_VAR,
    rng=None,
    early_stop=False,
) -> dict:
    """Compatibility API returning only the three would-KO features per option."""
    flags, _ = would_ko_flags_with_audit(
        obs,
        deck,
        enc,
        n_var=n_var,
        rng=rng,
        early_stop=early_stop,
    )
    return flags


def write_would_ko(obs, flags) -> None:
    """Write the engine-sim consequence onto each attack option in-place (the encoder emits these):
      o['would_ko']        KO/prize-take RATE in [0,1]
      o['would_ko_prizes'] EXPECTED prizes taken (0..6) -- grades a 2-prize ex KO above a 1-prize basic
      o['would_ko_win']    P(this move ENDS the game in my favour) in [0,1]"""
    opts = (obs.get("select") or {}).get("option") or []
    for i, r in flags.items():
        if 0 <= i < len(opts):
            ko, prizes, win = r
            opts[i]["would_ko"] = float(ko)
            opts[i]["would_ko_prizes"] = float(prizes)
            opts[i]["would_ko_win"] = float(win)


def annotate_would_ko(
    obs,
    deck,
    enc,
    n_var=WK_NDET_VAR,
    rng=None,
    early_stop=False,
) -> dict:
    """Compute would_ko_flags AND write them onto the attack options (the per-option TRAINING
    feature). Call ONCE per real (root) decision -- in the env at collection AND in the inference
    agent when net_config['would_ko'] -> train==test (both use the default n_var). Returns flags."""
    flags = would_ko_flags(
        obs,
        deck,
        enc,
        n_var=n_var,
        rng=rng,
        early_stop=early_stop,
    )
    write_would_ko(obs, flags)
    return flags


def annotate_would_ko_with_audit(
    obs,
    deck,
    enc,
    n_var=WK_NDET_VAR,
    rng=None,
    early_stop=False,
) -> tuple[dict, dict]:
    """Annotate options and return explicit validity/trial/failure metadata."""
    flags, audit = would_ko_flags_with_audit(
        obs,
        deck,
        enc,
        n_var=n_var,
        rng=rng,
        early_stop=early_stop,
    )
    write_would_ko(obs, flags)
    return flags, audit


def mcts_select(obs, net, enc, deck, tracker=None, ability=None,
                n_sims=160, n_det=2, c_puct=1.5, rng=None):
    """PUCT MCTS choice -> selection list[int] (net-greedy fallback / deck step)."""
    if obs.get("select") is None:                             # deck-selection step
        return [int(c) for c in deck]
    agg, ok = mcts_visits(obs, net, enc, deck, tracker, ability, n_sims, n_det, c_puct, rng)
    if not ok:
        return _net_greedy_select(obs, net, enc, deck, tracker, ability)
    return [int(agg.argmax())]


def mcts_policy(obs, net, enc, deck, tracker=None, ability=None,
                n_sims=160, n_det=2, c_puct=1.5, rng=None, temp=1.0):
    """For AlphaZero self-play on v2: returns (selection list[int], pi over N_ACTIONS)."""
    agg, ok = mcts_visits(obs, net, enc, deck, tracker, ability, n_sims, n_det, c_puct, rng)
    if not ok:
        return _net_greedy_select(obs, net, enc, deck, tracker, ability), None
    pi = np.zeros(N_ACTIONS, dtype=np.float32)
    if temp and temp != 1.0:
        agg = agg ** (1.0 / temp)
    pi[:len(agg)] = agg / agg.sum()
    a = int(np.random.choice(len(agg), p=pi[:len(agg)])) if temp > 0 else int(agg.argmax())
    return [a], pi
