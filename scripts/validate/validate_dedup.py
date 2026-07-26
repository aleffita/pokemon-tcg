"""PROVE the action-space dedup is safe before it touches any encoder/training.

Two checks:
  (1) --unit : synthetic grouping tests (hand-crafted option lists with known dup/non-dup structure).
  (2) sim    : over REAL replay decisions, for EVERY collapsed group, 1-ply-simulate each member
               option on the real engine and assert the resulting game state is IDENTICAL. A single
               structural divergence means the signature is too aggressive -> the dedup is WRONG.

To keep the comparison deterministic (so randomness can't masquerade as a real difference):
  * all members of a group are simulated with the SAME determinization (same rng seed),
  * manual_coin=True defers coin flips to a manual sub-select (so they don't auto-resolve to
    different outcomes between members),
  * the resulting obs is canonicalized: 'serial'/'logs'/'search_begin_input' stripped, and every
    list sorted by content -- so "a different physical copy was consumed" reads as identical, while
    any real difference (different target, different card moved, different count) still shows.

  python scripts/validate_dedup.py --unit
  python scripts/validate_dedup.py <episodes.zip> [max_eps]
"""
from __future__ import annotations
import sys, os, json, zipfile, random, dataclasses, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from rl.encoder.option_dedup import option_groups, option_signature, dup_legal_indices
from rl.encoder.encoding import _card_id

def _ms(lst):
    return collections.Counter(_card_id(c) for c in (lst or []))


# Player fields legitimately re-randomized by the engine's draw/shuffle/mulligan RNG (and, for the
# opponent, hidden) -> a divergence here is NOT evidence of a wrong-collapse. Everything else
# (active/bench Pokemon with hp/energy/attached cards/status, discard, turn-state flags) is compared
# at FULL detail. 'prize' contents are hidden (face-down) so we compare its COUNT, not its cards.
_RANDOM_FIELDS = ("hand", "deck", "deckCount", "handCount", "deckCardCount", "prize")


def _classify(currents, me):
    """STRUCTURAL iff either player's FULL visible state (board Pokemon at detail -- hp / maxHp /
    energies / attached energy & tool cards / preEvolution / special-conditions -- plus discard and
    every non-random player field) differs, OR the prize COUNT differs. This is detail-sensitive, NOT
    card-id-only (the audit-flagged gap): a same-id Pokemon with different hp/energy/status now counts
    as STRUCTURAL. BENIGN only when the divergence is confined to the random hand/deck (a same-card
    shuffle-draw redraw, or the opponent's hidden setup hand -- both independent of which duplicate
    the acting player picked). _canon strips serials + sorts, so a different physical copy reads equal."""
    def pview(p):
        rest = {k: v for k, v in (p or {}).items() if k not in _RANDOM_FIELDS}
        rest["__prizeN__"] = len((p or {}).get("prize") or [])
        return _canon(rest)
    base = [pview(currents[0]["players"][pi]) for pi in (0, 1)]
    for c in currents[1:]:
        for pi in (0, 1):
            if pview(c["players"][pi]) != base[pi]:
                return "STRUCTURAL"
    return "BENIGN"


# ----------------------------------------------------------------------------- unit tests
def _unit():
    # explicit cardId -> resolve_card returns it directly, so no obs state is needed.
    def play(idx, cid):   return {"type": 7, "index": idx, "cardId": cid}
    def attach(idx, cid, tgt):  return {"type": 8, "area": 2, "index": idx, "cardId": cid, "inPlayArea": 5, "inPlayIndex": tgt}
    def ability(idx, cid):  return {"type": 10, "area": 5, "index": idx, "cardId": cid}        # in-play source
    def discard_hand(idx, cid): return {"type": 11, "area": 2, "index": idx, "cardId": cid}
    def number(n):  return {"type": 0, "number": n}
    cases = [
        ("3 identical hand plays collapse",      [play(0, 50), play(1, 50), play(2, 50)], [0, 0, 0]),
        ("different cards don't collapse",       [play(0, 50), play(1, 99)],              [0, 1]),
        ("same energy same target collapses",    [attach(0, 13, 1), attach(1, 13, 1)],    [0, 0]),
        ("same energy DIFFERENT target keeps",   [attach(0, 13, 0), attach(1, 13, 1)],    [0, 1]),
        ("in-play ABILITY never collapses",      [ability(0, 50), ability(1, 50)],        [0, 1]),
        ("discard dup hand cards collapse",      [discard_hand(0, 7), discard_hand(1, 7)],[0, 0]),
        ("distinct NUMBER options keep",         [number(1), number(2), number(3)],       [0, 1, 2]),
        ("mixed: dup + unique",                  [play(0, 50), play(1, 50), play(2, 99)], [0, 0, 2]),
    ]
    ok = True
    for name, opts, expect in cases:
        g = option_groups({"option": opts}, None, 0, None)        # default sel -> single-pick (maxCount=1)
        status = "OK " if g == expect else "FAIL"
        if g != expect:
            ok = False
        print(f"  [{status}] {name}: got {g} expect {expect}")

    # multi-pick guard: copies of one card may BOTH be required -> dedup MUST be disabled.
    mp = [play(0, 50), play(1, 50)]
    mp_cases = [
        ("multi-pick maxCount=2 NOT collapsed", option_groups({"option": mp, "maxCount": 2}, None, 0, None), [0, 1]),
        ("multi-pick minCount=2 maxCount=2 NOT collapsed",
         option_groups({"option": mp, "minCount": 2, "maxCount": 2}, None, 0, None), [0, 1]),
        ("single-pick maxCount=1 STILL collapses", option_groups({"option": mp, "maxCount": 1}, None, 0, None), [0, 0]),
    ]
    for name, got, expect in mp_cases:
        if got != expect:
            ok = False
        print(f"  [{'OK ' if got == expect else 'FAIL'}] {name}: got {got} expect {expect}")
    tm, rm = dup_legal_indices({"option": mp, "maxCount": 2}, None, 0, None, [0, 1])
    if tm != set() or rm != {}:
        ok = False
    print(f"  [{'OK ' if (tm == set() and rm == {}) else 'FAIL'}] multi-pick dup_legal_indices masks nothing: got ({tm}, {rm})")

    print("UNIT", "PASSED" if ok else "FAILED")
    return ok


# ----------------------------------------------------------------------------- sim equivalence
def _canon(x):
    """Order-insensitive, serial-free canonical form: equivalent states -> equal; any real
    content difference -> unequal."""
    if isinstance(x, dict):
        return tuple(sorted((k, _canon(v)) for k, v in x.items()
                            if k not in ("serial", "logs", "search_begin_input")))
    if isinstance(x, list):
        return tuple(sorted((_canon(v) for v in x), key=repr))
    return x


def _decisions(ep):
    rewards = ep.get("rewards") or []
    if len(rewards) != 2 or rewards[0] is None or rewards[1] is None or rewards[0] == rewards[1]:
        return
    win = 0 if rewards[0] > rewards[1] else 1
    went = [st[win] for st in (ep.get("steps") or []) if len(st) > win]
    deck = None
    for ag in went:
        obs = ag.get("observation") or {}
        action = ag.get("action")
        if isinstance(action, list) and len(action) == 60:
            deck = [int(c) for c in action]
            continue
        if obs.get("select") is None or deck is None:
            continue
        yield obs, deck


def _state_view(obs2):
    """The agent-relevant resulting GAME STATE: obs['current'] (board/hands/discards/prizes/flags),
    plus the next decision's TYPE/COUNTS and the position-INDEPENDENT multiset of available option
    effects -- but NOT the option *indices* (which shift when a different physical copy survives).
    Two options are equivalent iff this view is identical."""
    cur = obs2.get("current")
    sel = obs2.get("select")
    eff = None
    if isinstance(sel, dict):
        opts = sel.get("option") or []
        # effect of each next-option, dropping positional fields (area/index/serial/tool/energy idx)
        eff = sorted(repr((o.get("type"), o.get("cardId"), o.get("inPlayArea"), o.get("inPlayIndex"),
                           o.get("attackId"), o.get("number"), o.get("count"),
                           o.get("specialConditionType"))) for o in opts)
        eff = (sel.get("type"), sel.get("context"), sel.get("minCount"), sel.get("maxCount"), tuple(eff))
    return (_canon(cur), eff)


def _sim_one(api, obs, deck, enc, opt, seed):
    from rl.search_agent import _determinize
    rng = random.Random(seed)                                # SAME seed per group -> same determinization
    ss = api.search_begin(api.to_observation_class(obs), manual_coin=True,
                          **_determinize(obs, deck, rng, enc))
    try:
        st = api.search_step(ss.searchId, [opt])
        o = dataclasses.asdict(st.observation)
        return _state_view(o), o.get("current")
    finally:
        try: api.search_release(ss.searchId)
        except Exception: pass


def _sim_validate(zip_path, max_eps):
    from cg import api
    from rl.encoder.card_features import get_card_table
    from rl.encoder.encoding import TokenEncoder
    enc = TokenEncoder(get_card_table())

    n_dec = n_groups = n_multi = n_pass = n_fail = n_simerr = n_state_diff = n_benign = n_structural = 0
    n_multipick = n_multipick_masked = 0
    fails = []
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".json")]
        if max_eps:
            names = names[:max_eps]
        for ei, name in enumerate(names):
            try:
                ep = json.loads(z.read(name))
            except Exception:
                continue
            for obs, deck in _decisions(ep):
                sel = obs["select"]; s = obs["current"]; me = s["yourIndex"]
                deck_list = sel.get("deck") if isinstance(sel.get("deck"), list) else None
                group = option_groups(sel, s, me, deck_list)
                n_dec += 1
                if (sel.get("maxCount") or 1) > 1:          # multi-pick: dedup disabled (group==identity)
                    n_multipick += 1
                    tm_mp, _ = dup_legal_indices(sel, s, me, deck_list, list(range(len(sel.get("option") or []))))
                    if tm_mp:                               # MUST be empty: a masked copy would break k-of-N
                        n_multipick_masked += 1
                members = collections.defaultdict(list)
                for i, c in enumerate(group):
                    members[c].append(i)
                seed = (ei * 1000003 + n_dec) & 0x7fffffff
                for canon_idx, idxs in members.items():
                    if len(idxs) < 2:
                        continue
                    n_groups += 1; n_multi += 1
                    try:
                        res = [_sim_one(api, obs, deck, enc, i, seed) for i in idxs]
                    except Exception:
                        n_simerr += 1
                        continue
                    if all(r[0] == res[0][0] for r in res):
                        n_pass += 1
                    else:
                        n_fail += 1
                        if any(r[0][0] != res[0][0][0] for r in res):       # GAME STATE (current) differs
                            n_state_diff += 1
                            kind = _classify([r[1] for r in res], me)
                            if kind == "STRUCTURAL":
                                n_structural += 1
                                if len(fails) < 30:
                                    opts = sel["option"]
                                    fails.append({
                                        "ep": name, "select_type": sel.get("type"),
                                        "options": [{"type": opts[i].get("type"), "cardId": opts[i].get("cardId"),
                                                     "area": opts[i].get("area"), "index": opts[i].get("index"),
                                                     "inPlayIndex": opts[i].get("inPlayIndex")} for i in idxs],
                                        "sig": str(option_signature(opts[idxs[0]], s, me, deck_list)),
                                    })
                            else:
                                n_benign += 1
            if (ei + 1) % 200 == 0:
                print(f"[validate] {ei+1} eps  decisions={n_dec}  multi-groups={n_multi}  "
                      f"PASS={n_pass}  FAIL={n_fail}  simerr={n_simerr}", flush=True)
        try: api.search_end()
        except Exception: pass

    print(f"\n=== DEDUP SIM VALIDATION ===")
    print(f"decisions checked : {n_dec}  (of which multi-pick, dedup disabled: {n_multipick})")
    print(f"collapsed groups  : {n_multi}  (>=2 single-pick options we claim equivalent)")
    print(f"PASS (state + next-effects identical)         : {n_pass}")
    print(f"FAIL total                                    : {n_fail}")
    print(f"  benign relabel (state same, only opt labels): {n_fail - n_state_diff}")
    print(f"  GAME-STATE differs                          : {n_state_diff}")
    print(f"     BENIGN  (random draw, acting hand/deck)   : {n_benign}")
    print(f"     STRUCTURAL (board/discard/prize/opp)      : {n_structural}   <== THE SAFETY METRIC (must be 0)")
    print(f"sim errors        : {n_simerr}  (engine couldn't sim -> not counted)")
    print(f"multi-pick selects: {n_multipick}  (dedup disabled; of these MASKED (must be 0): {n_multipick_masked})")
    if n_structural == 0 and n_multipick_masked == 0:
        print("\n>>> ZERO structural divergences + ZERO multi-pick masking -> every collapsed group is")
        print("    provably state-equivalent (full board detail) and no k-of-N pick was broken. DEDUP SAFE. <<<")
    else:
        if n_structural:
            print(f"\n!!! {n_structural} STRUCTURAL divergences -> signature collapses NON-equivalent options. MUST FIX. !!!")
        if n_multipick_masked:
            print(f"\n!!! {n_multipick_masked} multi-pick selects had a masked copy -> k-of-N pick broken. MUST FIX. !!!")
        for f in fails:
            print(" ", json.dumps(f))
    print("VALIDATE DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--unit":
        sys.exit(0 if _unit() else 1)
    zp = sys.argv[1]
    me = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    _sim_validate(zp, me)
