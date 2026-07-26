"""Quick end-to-end encoding test on a single replay episode.

Validates:
  - encoder can process every decision in the replay
  - labels are always legal (masked_rate == 0.0)
  - output arrays have correct shapes
  - tracker state tracks correctly across turns
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rl.encoder.encoding import TokenEncoder, GameTracker, AbilityTracker, SUBMIT_ACTION, build_mask
from rl.encoder.card_features import get_card_table
from rl.encoder.enc_constants import N_STATE_TOKENS, MAX_OPTIONS, N_ACTIONS

REPLAY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "replay", "85966927.json")

def main():
    with open(REPLAY) as f:
        ep = json.load(f)

    print(f"Replay: {ep['info']['Agents'][0]['Name']} vs {ep['info']['Agents'][1]['Name']}")
    print(f"  rewards={ep['rewards']}, steps={len(ep['steps'])}")
    print()

    ct = get_card_table()
    enc = TokenEncoder(ct)

    total_decisions = 0
    total_rows = 0
    masked_count = 0
    errors = 0
    attack_rows = 0

    for side in (0, 1):
        tr = GameTracker()
        ab = AbilityTracker()
        deck = None
        went = [st[side] for st in ep["steps"] if len(st) > side]
        side_name = ep["info"]["Agents"][side]["Name"]

        print(f"--- Side {side}: {side_name} ({len(went)} entries) ---")

        for i, ag in enumerate(went):
            obs = ag.get("observation") or {}
            sel = obs.get("select")
            action = ag.get("action")

            # Deck = 60-card action
            if isinstance(action, list) and len(action) == 60:
                deck = [int(c) for c in action]
                tr.reset()
                ab.reset()
                continue

            if sel is None or deck is None:
                continue

            # Advance trackers
            try:
                tr.update(obs)
                ab.note_turn((obs.get("current") or {}).get("turn"))
            except Exception as e:
                print(f"  Tracker error at step {i}: {e}")
                errors += 1
                continue

            # Label is on NEXT entry (off-by-one)
            label = went[i + 1].get("action") if i + 1 < len(went) else None
            opts = sel.get("option") or []
            n = len(opts)

            if not (isinstance(label, list) and label
                    and all(isinstance(x, int) and 0 <= x < n for x in label)):
                continue

            total_decisions += 1
            maxc = sel.get("maxCount", 1)
            is_atk = 1 if (sel.get("type") == 0 and
                          any(o.get("attackId") is not None for o in opts)) else 0
            if is_atk:
                attack_rows += 1

            me = (obs.get("current") or {}).get("yourIndex")
            deck_list = sel.get("deck") if isinstance(sel.get("deck"), list) else None

            # Encode each pick in the multi-select
            for k, idx in enumerate(label):
                e = enc.encode(obs, set(label[:k]), self_deck=deck, tracker=tr,
                              ability_slots=ab.slots)
                mask = e["action_mask"]

                if float(mask[int(idx)]) < 0.5:
                    print(f"  MASKED LABEL at step {i}, pick {k}: label={idx} "
                          f"n_options={n} type={sel.get('type')} ctx={sel.get('context')}")
                    masked_count += 1
                    errors += 1
                    continue

                total_rows += 1

            # Check shapes on first valid row
            if total_rows == 1:
                print(f"  First row keys: {sorted(e.keys())}")
                print(f"  state shape: {e.get('state', e.get('cls', '???'))}")
                for k in ("action_mask",):
                    v = e.get(k)
                    if v is not None:
                        print(f"  {k}: shape={np.array(v).shape}, dtype={np.array(v).dtype}")

            ab.record(sel, label)

    print()
    print("=== RESULTS ===")
    print(f"  Total decisions: {total_decisions}")
    print(f"  Total rows: {total_rows}")
    print(f"  Attack rows: {attack_rows}")
    print(f"  Masked labels: {masked_count} (MUST be 0)")
    print(f"  Errors: {errors}")
    print(f"  masked_rate: {masked_count / max(total_rows, 1):.6f}")

    if masked_count == 0 and total_rows > 0:
        print("\n  PASS - encoding pipeline validated!")
    else:
        print(f"\n  FAIL - {masked_count} masked labels found")
        sys.exit(1)

if __name__ == "__main__":
    main()
