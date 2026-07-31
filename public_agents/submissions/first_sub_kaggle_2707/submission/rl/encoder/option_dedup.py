"""Action-space DEDUP: collapse PROVABLY-interchangeable options into one canonical choice so the
policy doesn't split probability / entropy across identical options (the redundancy exposed by the
BC equivalence analysis: e.g. attaching any of 3 copies of the same energy, discarding any of N
identical hand cards -- same effect, different option index).

SAFETY-FIRST / CONSERVATIVE by design. We ONLY ever collapse copies of the same STATELESS card
(HAND / DECK / LOOKING zones) that match on every effect-relevant field and differ ONLY in which
physical copy is used. We NEVER collapse:
  * in-play entities (active/bench) -- two same-species Pokemon can differ in hp/energy/state;
  * attached-tool / attached-energy selects (OptionType 4/5/6) -- involve in-play state;
  * options with different targets, attacks, counts, special-conditions, or owners;
  * ANY option inside a MULTI-PICK (maxCount>1) select -- a 'discard/search/put N' effect may need
    BOTH physical copies as distinct indices, so collapsing them makes the k-of-N pick unreachable.
When anything is uncertain (unresolved source card, non-stateless source) we DO NOT dedup.

This is the HYPOTHESIS; correctness is *proven empirically* by scripts/validate_dedup.py, which
1-ply-simulates every option in every collapsed group on the real engine and asserts the resulting
game states are identical. Do NOT wire this into the encoder / BC / RL until that validation passes.
"""
from __future__ import annotations

from .encoding import resolve_card

# Stateless source zones: a card here has no in-play state, so two copies of the same card id are
# truly interchangeable. (AreaType: DECK=1, HAND=2, LOOKING=12.)
_STATELESS_SRC = frozenset({1, 2, 12})
# Option types whose source is an ATTACHED card/energy on an in-play Pokemon -> never dedup.
_ATTACHED_SELECT = frozenset({4, 5, 6})


def _multipick(sel: dict) -> bool:
    """A select that may consume MORE THAN ONE option (maxCount > 1). For such selects two copies of
    the same stateless card are NOT interchangeable in aggregate: a 'discard 2 / search 2 / put 2'
    effect needs BOTH physical copies as DISTINCT indices (the engine rejects a repeated index and
    rejects a too-short pick). Collapsing them would make the required k-of-N selection unreachable,
    so we DISABLE dedup entirely for any maxCount>1 select. Dedup is only sound for single-pick
    (maxCount==1) selects, where exactly one copy is consumed and any copy yields the same effect."""
    return (sel.get("maxCount") or 1) > 1


def option_signature(o: dict, s, me: int, deck_list):
    """Hashable equivalence key for one option, or ``None`` to mark it NEVER-dedup (kept unique).

    Two options sharing a non-None key are *claimed* interchangeable: same verb, same source card
    id, same owner, same target, same attack/number/count/special-condition -- differing only in
    which physical copy of a stateless (hand/deck/looking) card is consumed. The source ``index`` /
    ``serial`` (the copy selector) is deliberately EXCLUDED from the key.

    Returns None (never collapse) when: the option selects an attached card/energy (4/5/6); the
    source is not a stateless zone (in-play, stadium, unknown); or the source card can't be resolved.
    """
    ot = o.get("type")
    if ot in _ATTACHED_SELECT:                  # attached tool/energy on a Pokemon -> in-play state
        return None
    area = o.get("area")
    if area is None and ot == 7:                # PLAY carries a bare HAND index, no explicit area
        area = 2
    if area not in _STATELESS_SRC:              # in-play / stadium / unknown source -> never collapse
        return None
    src = resolve_card(o, s, me, deck_list)
    if not src:                                 # could not resolve the source card -> do not risk it
        return None
    return (ot, int(area), int(src), o.get("playerIndex"),
            o.get("inPlayArea"), o.get("inPlayIndex"), o.get("attackId"),
            o.get("number"), o.get("count"), o.get("specialConditionType"))


def option_groups(sel: dict, s, me: int, deck_list):
    """``group[i]`` = canonical option index for option ``i`` (== ``i`` if it is a canonical or is
    never-dedup). Options sharing a signature map to the FIRST occurrence. Operates over ALL options
    (ignores legality) -- used by the validation harness to check every collapsed group."""
    options = sel.get("option") or []
    group = list(range(len(options)))
    if _multipick(sel):                         # k-of-N select: every copy may be REQUIRED -> no dedup
        return group
    canon: dict = {}
    for i, o in enumerate(options):
        sig = option_signature(o, s, me, deck_list)
        if sig is None:
            continue                            # unique / never-dedup
        c = canon.get(sig)
        if c is None:
            canon[sig] = i                      # first with this signature is the canonical
        else:
            group[i] = c
    return group


def dup_legal_indices(sel: dict, s, me: int, deck_list, legal_idx):
    """Among the LEGAL option indices (in order), return ``(to_mask, remap)`` where ``to_mask`` is
    the set of non-canonical duplicate indices to mask OUT (keeping the FIRST legal option of each
    signature group selectable), and ``remap[i]`` maps a masked dup ``i`` -> its canonical legal
    index (for remapping a BC expert label onto the surviving option). Legality-aware so every group
    keeps a legal representative. This is the hook for encoder / BC integration (later)."""
    options = sel.get("option") or []
    canon: dict = {}
    to_mask: set = set()
    remap: dict = {}
    if _multipick(sel):                         # k-of-N select: copies are not aggregate-interchangeable
        return to_mask, remap                   # -> mask nothing, remap nothing (leave the select intact)
    for i in legal_idx:                         # only legal options, in their natural order
        sig = option_signature(options[i], s, me, deck_list)
        if sig is None:
            continue
        c = canon.get(sig)
        if c is None:
            canon[sig] = i                      # first LEGAL with this signature stays legal
        else:
            to_mask.add(i)
            remap[i] = c
    return to_mask, remap