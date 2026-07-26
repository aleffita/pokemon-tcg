"""CleanRL-style PPO for the cabt env, with action masking and self-play.

Curriculum: train vs the random opponent until ``--selfplay-start`` env steps,
then switch the in-worker opponent to frozen snapshots of the learner (a small
pool, refreshed every ``--snapshot-every`` steps).

Run:
    python -m rl.train --total-timesteps 5000000 --num-envs 16 --device cuda
The engine is CPU-bound and one-battle-per-process, so throughput scales with
--num-envs (subprocess workers); the net is tiny and barely uses the GPU.
"""

from __future__ import annotations

import argparse
import collections
import os
import threading
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from rl.encoder.card_features import get_card_table
from rl.deck.decks import DECKS
try:
    from rl.deck.decks_generated import GENERATED      # auto-generated archetype decks
except Exception:
    GENERATED = {}
from rl.encoder.encoding import TokenEncoder
from rl.env.env import load_deck
from .policy import build_token_net, obs_to_tensors
from rl.env.vec_env import SubprocVecEnv
from .lr_schedule import lr_at


def resolve_deck_pool(name: str) -> list[list[int]]:
    """Map a --decks value to a list of decks (each side is sampled from this)."""
    sample = load_deck()  # engine sample deck (agent/deck.csv)
    if name == "all":
        return list(DECKS.values()) + [sample]
    if name in ("gen", "all+gen") and not GENERATED:   # import failed -> don't silently shrink the pool
        raise SystemExit("--decks gen/all+gen but rl/decks_generated.py is missing/empty "
                         "(run scripts/build_decks.py)")
    if name == "gen":                                  # 50 generated archetypes
        return list(GENERATED.values())
    if name == "all+gen":                              # official + sample + generated
        return list(DECKS.values()) + [sample] + list(GENERATED.values())
    if name == "official":
        return list(DECKS.values())
    if name == "sample":
        return [sample]
    if name in ("good", "real", "meta"):               # 4 official + real (Limitless + Kaggle-mined) decks
        try:
            from rl.deck.decks_meta import META                # META self-merges decks_kaggle (see decks_meta.py)
        except Exception:
            META = {}
        return list(DECKS.values()) + list(META.values())
    if name in ("top15", "train15"):                   # top-15 Kaggle archetypes + 7 CL Aichi (decks_train.py)
        from rl.deck.decks_train import TRAIN_TOP15
        return list(TRAIN_TOP15.values())
    if name in ("top50", "train50"):                   # top-50 Kaggle archetypes + 7 CL Aichi
        from rl.deck.decks_train import TRAIN_TOP50
        return list(TRAIN_TOP50.values())
    if name in DECKS:
        return [DECKS[name]]
    if name in GENERATED:
        return [GENERATED[name]]
    raise SystemExit(f"unknown --decks '{name}' "
                     "(all|gen|all+gen|official|sample|good|top15|top50|<name>)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--anneal-lr", action="store_true", default=True)
    p.add_argument("--lr-schedule", choices=["linear", "cosine"], default="cosine")
    p.add_argument("--warmup-steps", type=int, default=2_000_000,
                   help="linear LR warmup over this many GLOBAL env steps (flat, budget-invariant; "
                        "clamped to 20%% of the run; matches train_selfplay)")
    p.add_argument("--warmup-frac", type=float, default=0.0,
                   help="DEPRECATED %%-of-decay-window warmup; >0 overrides --warmup-steps")
    p.add_argument("--lr-min-ratio", type=float, default=0.1,
                   help="final LR = lr * this (schedule floor). Default 0.1: a ~0 floor freezes the "
                        "tail of the run (kl->0) at full collection cost")
    p.add_argument("--decay-steps", type=int, default=None,
                   help="LR decays over this many GLOBAL steps then holds FLAT at lr*min_ratio; "
                        "None = 70%% of --total-timesteps")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--target-kl", type=float, default=None,
                   help="KL early-stop: stop the update epochs once this iter's approx-KL exceeds this "
                        "(e.g. 0.03) -> bounds the trust region (recommended for d256, which runs hot)")
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--vf-clip", type=float, default=None,
                   help="PPO value-clip range (SCALAR head only; ignored when --value-categorical): a "
                        "trust region on the value update, max(unclipped, clipped) MSE. None=off.")
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--norm-adv", action="store_true", default=True)
    # self-play
    p.add_argument("--selfplay-start", type=int, default=500_000,
                   help="env steps of random-opponent warmup before self-play")
    p.add_argument("--snapshot-every", type=int, default=100_000)
    p.add_argument("--pool-size", type=int, default=5)
    # teacher-student stabilization (B6): distil toward a frozen last-best (anti-regression).
    p.add_argument("--teacher-kl-coef", type=float, default=0.0,
                   help="teacher-student: weight on KL(last-best policy || student policy) (0=off)")
    p.add_argument("--teacher-value-coef", type=float, default=0.0,
                   help="teacher-student: weight on value distillation to last-best (CE if categorical "
                        "else MSE) (0=off)")
    # self-play loop (C): pure-live opponent + gated teacher promotion.
    p.add_argument("--pure-live", action="store_true",
                   help="C10: opponent = the CURRENT live net (refreshed each iter) instead of a frozen "
                        "snapshot pool; the teacher (--teacher-*) provides stability instead of the pool")
    p.add_argument("--promote-winrate", type=float, default=None,
                   help="C11: only promote the last-best/teacher when a fresh snapshot beats it by >= this "
                        "summed-both-orientation winrate (None=time-based refresh every --snapshot-every)")
    p.add_argument("--promote-games", type=int, default=200,
                   help="C11: games (summed both orientations) for the promotion eval")
    p.add_argument("--two-sided", action="store_true",
                   help="C9: collect BOTH seats' decisions per game (the live net plays both sides)")
    # env / data
    p.add_argument("--shaping", choices=["none", "prize_diff"], default="none")
    p.add_argument("--terminal-margin", type=float, default=0.0,
                   help="F19 graded terminal reward: R = sign*(1 + alpha*loser_prizes_remaining/6); "
                        "0 = flat +/-1. If combined with --value-categorical, set --value-vmax >= 1+alpha.")
    p.add_argument("--decks", type=str, default="all",
                   help="deck pool: 'all' (4 official + sample), 'official' (4), 'sample', "
                        "or a named deck from rl.decks (e.g. mega_abomasnow)")
    p.add_argument("--no-randomize-side", action="store_true")
    # net (architecture)
    p.add_argument("--arch", default="transformer2")  # single token-transformer encoding (effect-features folded in)
    p.add_argument("--static", action="store_true", help="transformer2: feed static per-card features (HP/type/cost/...) into the net")
    p.add_argument("--split-heads", action="store_true", help="transformer2: dedicated submit+value tokens instead of both reading CLS (value-head decoupling)")
    # distributional (categorical) value head + register tokens
    p.add_argument("--value-categorical", action="store_true",
                   help="distributional value head: cross-entropy over atoms instead of scalar MSE "
                        "(better-conditioned under noisy/non-stationary targets; aimed at generalist value regression)")
    p.add_argument("--value-atoms", type=int, default=51, help="categorical value: number of support atoms")
    p.add_argument("--value-vmax", type=float, default=1.0,
                   help="categorical value support = [-vmax, vmax]; raise above 1.0 if --terminal-margin "
                        "pushes returns past +/-1")
    p.add_argument("--would-ko", action=argparse.BooleanOptionalAction, default=True,
                   help="transformer2: annotate engine-simulated would-KO per attack option (DEFAULT ON; "
                        "--no-would-ko to disable). The env runs a 1-ply SDK sim per attack; net learns lethality.")
    p.add_argument("--native-encode", action="store_true",
                   help="collection speedup: run the worker envs on the cg engine with the Cython "
                        "encode_native (byte-identical to the Python encode). ~1.9x collection with --no-would-ko "
                        "(JSON-round-trip bound; ~1.16x with would_ko since sims dominate). Requires the built "
                        "encode_native module + cg/libcg_custom.so. Training-only; inference is unchanged.")
    p.add_argument("--obs-shm", action="store_true",
                   help="workers write encoded rollout obs into a shared-memory batch buffer instead of "
                        "pickling over the Pipe (main reads zero-copy) -> cuts collection IPC + np.stack.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the UPDATE forward (b>1, static shapes) to fuse the memory-bound "
                        "d128 matmuls; collection stays uncompiled (its b=1 path is dynamic).")
    p.add_argument("--bf16", action="store_true",
                   help="run the PPO-update forward under bf16 autocast (collection stays fp32); loss upcast "
                        "to fp32. Speeds the memory-bound d128 update (TF32 barely helps -- not FLOP-bound).")
    # mlp dims
    p.add_argument("--card-h", type=int, default=64)
    p.add_argument("--trunk-h", type=int, default=256)
    p.add_argument("--opt-h", type=int, default=96)
    # transformer dims
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--nlayers", type=int, default=3)
    p.add_argument("--ff", type=int, default=256)
    p.add_argument("--init-from", type=str, default=None,
                   help="checkpoint .pt to WARM-START net weights from (fresh optimizer + LR -> aggressive)")
    p.add_argument("--resume", type=str, default=None,
                   help="checkpoint .pt (a latest.pt) to FULLY resume: net + optimizer + global_step, so "
                        "the LR schedule + Adam momentum continue with NO jump (use this to continue a run)")
    # infra
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--server-device", type=str, default="cpu",
                   help="device for the batched opponent inference server (cpu safe; cuda faster)")
    p.add_argument("--opponent-mode", choices=["local", "server"], default="local",
                   help="local: opponent net runs per-worker on CPU; server: opponent forward is "
                        "batched in a main-process GPU thread (defaults server-device to --device)")
    p.add_argument("--scripted-opponent", type=str, default=None,
                   help="path to a Kaggle-style main.py (def agent(obs)->list[int]) used as a FIXED "
                        "opponent (no self-play). Its sibling deck.csv becomes the opponent deck. "
                        "Train our --decks agent FROM SCRATCH against this external agent.")
    p.add_argument("--async-collect", action="store_true",
                   help="overlap the PPO update (GPU) with collection of the NEXT rollout on a background "
                        "thread, using a frozen behavior-net snapshot (1-update-stale; clip absorbs it)")
    p.add_argument("--collector-device", type=str, default=None,
                   help="--async-collect: run the collector's behavior-net forwards on THIS device "
                        "(e.g. cuda:1) so they don't contend with the learner update on --device")
    p.add_argument("--ddp", action="store_true",
                   help="data-parallel across processes (launch with torchrun --nproc_per_node=N); each rank "
                        "collects+updates, gradients all-reduced -> ~Nx throughput, no staleness, Nx batch")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=str, default=os.path.join(os.environ.get("HOME", "."), "pkmn_runs", "ppo"))
    p.add_argument("--save-every", type=int, default=500_000)
    p.add_argument("--log-every", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches

    if args.value_categorical and args.terminal_margin > 0 and args.value_vmax < 1.0 + args.terminal_margin:
        raise SystemExit(f"--value-vmax ({args.value_vmax}) < 1+terminal_margin ({1.0 + args.terminal_margin}): "
                         f"graded returns would be clamped, miscalibrating the categorical value targets. "
                         f"Set --value-vmax >= {1.0 + args.terminal_margin}.")

    # C9/C11 are reserved but NOT implemented yet -- fail loudly rather than silently no-op.
    if args.two_sided:
        raise SystemExit("--two-sided (C9 two-sided/all-seats collection) is not implemented yet: it "
                         "restructures the env step protocol (env must yield BOTH seats' decisions). "
                         "Run without it.")
    if args.promote_winrate is not None:
        raise SystemExit("--promote-winrate (C11 gated teacher promotion) is not implemented yet: it "
                         "needs an in-process both-orientation h2h eval. The teacher currently refreshes "
                         "on a timer (every --snapshot-every); run without --promote-winrate for that.")

    # ---- DDP (data-parallel) setup; world=1 when not --ddp ----
    ddp = args.ddp
    if ddp:
        dist.init_process_group(backend="nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world, local_rank = 0, 1, 0
        device = torch.device(args.device)
    is_main = rank == 0
    collector_device = torch.device(args.collector_device) if args.collector_device else device

    if is_main:
        os.makedirs(args.out, exist_ok=True)
        print(f"[cfg] {vars(args)}", flush=True)
        if ddp:
            print(f"[ddp] world={world} -> effective batch {args.batch_size * world}", flush=True)

    torch.manual_seed(args.seed + rank)             # per-rank seeds -> diverse envs/opponents
    np.random.seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True    # TF32 tensor cores for fp32 matmuls (Ampere+/Blackwell):
    torch.backends.cudnn.allow_tf32 = True           # large matmul speedup at negligible training-quality cost

    ct = get_card_table()
    net_config = {"arch": args.arch, "d_model": args.d_model,
                  "nhead": args.nhead, "nlayers": args.nlayers, "ff": args.ff,
                  "static": args.static, "would_ko": args.would_ko,
                  "split_heads": args.split_heads,
                  "value_categorical": args.value_categorical,
                  "value_atoms": args.value_atoms, "value_vmax": args.value_vmax}
    enc = TokenEncoder(ct)
    to_tensors = obs_to_tensors

    pool = resolve_deck_pool(args.decks)
    print(f"[decks] pool='{args.decks}' -> {len(pool)} deck(s); both sides sampled per episode", flush=True)
    env_kwargs = {
        "agent_decks": pool,
        "opponent_decks": pool,
        "randomize_side": not args.no_randomize_side,
        "shaping": args.shaping,
        "would_ko": args.would_ko,
        "terminal_margin": args.terminal_margin,
        "native_encode": args.native_encode,
    }
    if args.scripted_opponent:                               # FIXED external opponent -> no self-play
        deck_dir = os.path.dirname(os.path.abspath(args.scripted_opponent))
        with open(os.path.join(deck_dir, "deck.csv")) as f:
            b_deck = [int(x.strip()) for x in f if x.strip()]
        assert len(b_deck) == 60, f"scripted opponent deck has {len(b_deck)} cards (need 60)"
        env_kwargs["opponent_decks"] = [b_deck]              # opponent always plays its own deck
        env_kwargs["scripted_opponent"] = os.path.abspath(args.scripted_opponent)
        args.opponent_mode = "local"                         # scripted runs per-worker; no inference server
        args.selfplay_start = 10 ** 18                       # never switch to self-play -> opponent stays fixed
        if is_main:
            print(f"[scripted-opp] FIXED opponent {args.scripted_opponent} (deck {len(b_deck)} cards); "
                  f"agent decks={args.decks}; self-play DISABLED", flush=True)

    srv_dev = args.server_device
    if args.opponent_mode == "server" and srv_dev == "cpu":
        srv_dev = args.device                                # default the opponent server to the GPU
    envs = SubprocVecEnv(args.num_envs, env_kwargs, net_config, base_seed=args.seed * 1000,
                         server_device=srv_dev, opponent_mode=args.opponent_mode,
                         obs_shm=args.obs_shm)

    net = build_token_net(ct, net_config).to(device)
    if ddp:                                              # identical initial weights across ranks
        for p in net.parameters():
            dist.broadcast(p.data, src=0)
        for b in net.buffers():
            dist.broadcast(b.data, src=0)
    opt = optim.AdamW(net.parameters(), lr=args.lr, eps=1e-5, weight_decay=0.0,
                      fused=(device.type == "cuda"))   # wd=0 -> identical math to Adam; fused CUDA kernel
    envs.pin_obs_buffers()                             # DMA H2D for the shm obs (best-effort; needs CUDA up)
    if is_main:
        print(f"[net] params={sum(p.numel() for p in net.parameters()):,}", flush=True)

    if args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        net.load_state_dict(ck["net"])                  # STRICT: any arch mismatch raises
        print(f"[init] loaded {args.init_from} (was trained to step {ck.get('global_step')})", flush=True)

    resume_step = 0
    if args.resume:                                          # FULL resume: net + optimizer + step (no LR jump)
        rck = torch.load(args.resume, map_location=device)
        net.load_state_dict(rck["net"])                      # STRICT: any arch mismatch raises
        if rck.get("opt") is not None:
            opt.load_state_dict(rck["opt"])
            for st in opt.state.values():                    # move optimizer state to this device
                for k, v in st.items():
                    if isinstance(v, torch.Tensor):
                        st[k] = v.to(device)
        resume_step = int(rck.get("global_step", 0))
        if is_main:
            print(f"[resume] {args.resume}: net+optimizer restored at step {resume_step} "
                  f"(LR schedule + Adam momentum continue, no jump)", flush=True)

    # Optional torch.compile of the UPDATE forward ONLY (minibatches are b>1 with static shapes ->
    # compile-friendly). Collection keeps the uncompiled net.logits_value: its b=1 path does a
    # data-dependent nonzero() truncation that would thrash the compiler. Shares net's params by
    # reference, so DDP grad all-reduce + opt.step() (in-place param updates) work unchanged.
    update_lv = net.logits_value
    if args.compile:
        update_lv = torch.compile(net.logits_value, dynamic=False)
        if is_main:
            print("[compile] update forward compiled (torch.compile dynamic=False)", flush=True)

    # teacher-student (B6): a frozen last-best the student is distilled toward (anti-regression). The
    # teacher is refreshed to the current net at the snapshot cadence (gated promotion = a future
    # in-process h2h eval; see --promote-winrate). Provides the stability the snapshot pool gave, so it
    # pairs with --pure-live (C10), where the pool is dropped and the opponent is the latest current net.
    use_teacher = args.teacher_kl_coef > 0.0 or args.teacher_value_coef > 0.0
    teacher_net = None
    if use_teacher:
        teacher_net = build_token_net(ct, net_config).to(device)
        teacher_net.load_state_dict(net.state_dict()); teacher_net.eval()
        for tp in teacher_net.parameters():
            tp.requires_grad_(False)
        if is_main:
            print(f"[teacher] B6 on (kl={args.teacher_kl_coef} value={args.teacher_value_coef}); "
                  f"refresh every {args.snapshot_every} steps"
                  + ("" if args.promote_winrate is None else f"; GATED >= {args.promote_winrate} winrate"),
                  flush=True)

    def refresh_teacher():
        if teacher_net is not None:
            teacher_net.load_state_dict(net.state_dict()); teacher_net.eval()

    shapes = enc.shapes
    _PROF = os.environ.get("TRAIN_PROF") == "1"

    def make_buffers():
        """One rollout-storage set (obs kept on device). Two are allocated for --async-collect."""
        obs_buf = {}
        for k, sh in shapes.items():
            dt = torch.long if k in enc.int_keys else torch.float32
            obs_buf[k] = torch.zeros((args.num_steps, args.num_envs, *sh), dtype=dt, device=device)
        return {"obs": obs_buf,
                "act": torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device),
                "logp": torch.zeros((args.num_steps, args.num_envs), device=device),
                "rew": torch.zeros((args.num_steps, args.num_envs), device=device),
                "done": torch.zeros((args.num_steps, args.num_envs), device=device),
                "val": torch.zeros((args.num_steps, args.num_envs), device=device)}

    def collect_rollout(behavior, start_obs, start_done, buf, prof_sync, fwd_device):
        """Fill `buf` (on `device`) with one rollout stepped by `behavior`; return end-state + bootstrap.
        Shared by sync (behavior=net, fwd_device=device) and async (frozen clone). For 2-GPU async
        fwd_device!=device: the forward runs on fwd_device (no contention with the learner) and only the
        small action/value tensors cross back to `device`; the rollout buffer always lives on `device`."""
        obs_buf = buf["obs"]
        cur_obs, cur_done = start_obs, start_done
        two_dev = fwd_device != device
        eprs = []
        if prof_sync and device.type == "cuda":
            torch.cuda.synchronize()
        tc0 = time.time()
        for step in range(args.num_steps):
            for k in shapes:
                obs_buf[k][step] = cur_obs[k]
            buf["done"][step] = cur_done
            fobs = {k: v.to(fwd_device) for k, v in cur_obs.items()} if two_dev else cur_obs
            with torch.no_grad():
                action, logp, _, value = behavior.get_action_and_value(fobs)
            if two_dev:
                action, logp, value = action.to(device), logp.to(device), value.to(device)
            buf["act"][step] = action
            buf["logp"][step] = logp
            buf["val"][step] = value
            next_obs_np, reward, done, infos = envs.step(action.cpu().numpy())
            buf["rew"][step] = torch.as_tensor(reward, device=device)
            cur_obs = to_tensors(next_obs_np, device)
            cur_done = torch.as_tensor(done, dtype=torch.float32, device=device)
            for d, info in zip(done, infos):
                if d and "terminal_reward" in info:
                    eprs.append(info["terminal_reward"])
        with torch.no_grad():
            bobs = {k: v.to(fwd_device) for k, v in cur_obs.items()} if two_dev else cur_obs
            boot_val = behavior.get_value(bobs)
            if two_dev:
                boot_val = boot_val.to(device)
        if prof_sync and device.type == "cuda":
            torch.cuda.synchronize()
        return {"end_obs": cur_obs, "end_done": cur_done, "boot_val": boot_val,
                "eprs": eprs, "nsteps": args.num_steps * args.num_envs, "tc": time.time() - tc0}

    def compute_gae(buf, boot_val, end_done):
        with torch.no_grad():
            rewards, values, dones = buf["rew"], buf["val"], buf["done"]
            advantages = torch.zeros_like(rewards)
            lastgae = 0
            for t in reversed(range(args.num_steps)):
                nonterminal = 1.0 - (end_done if t == args.num_steps - 1 else dones[t + 1])
                nextval = boot_val if t == args.num_steps - 1 else values[t + 1]
                delta = rewards[t] + args.gamma * nextval * nonterminal - values[t]
                advantages[t] = lastgae = delta + args.gamma * args.gae_lambda * nonterminal * lastgae
            return advantages, advantages + values

    def update_net(buf, advantages, returns, teacher=None):
        """K-epoch minibatch PPO update on `net`. Returns diagnostics for logging.
        `teacher` (optional, B6): a frozen last-best net whose policy + value are distilled into the
        student (anti-regression) via --teacher-kl-coef / --teacher-value-coef."""
        b_obs = {k: buf["obs"][k].reshape((-1, *shapes[k])) for k in shapes}
        b_logp = buf["logp"].reshape(-1)
        b_actions = buf["act"].reshape(-1)
        b_adv = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_vals = buf["val"].reshape(-1)                        # collection-time values (for value clipping)
        # teacher-student (B6): cache the frozen last-best's policy logits + value ONCE per iter
        # (chunked, no_grad) so every update minibatch can add KL + value-distill without re-running it.
        t_cache = None
        if teacher is not None and (args.teacher_kl_coef > 0.0 or args.teacher_value_coef > 0.0):
            tl, tv, tvd = [], [], []
            with torch.no_grad():
                for s in range(0, args.batch_size, args.minibatch_size):
                    cmb = {k: b_obs[k][s:s + args.minibatch_size] for k in shapes}
                    cl, cv, cvd = teacher.logits_value(cmb, return_vdist=True)
                    tl.append(cl); tv.append(cv)
                    if cvd is not None:
                        tvd.append(cvd)
            t_cache = {"logits": torch.cat(tl), "val": torch.cat(tv),
                       "vdist": torch.cat(tvd) if tvd else None}
        inds = np.arange(args.batch_size)
        clipfracs, approx_kls = [], []
        last = {"pg": 0.0, "v": 0.0, "ent": 0.0}
        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for s in range(0, args.batch_size, args.minibatch_size):
                mb = inds[s:s + args.minibatch_size]
                mb_obs = {k: b_obs[k][mb] for k in shapes}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                    s_logits, newval, vdist = update_lv(mb_obs, return_vdist=True)
                if args.bf16:                                   # upcast -> loss/ratios/softmax in fp32 (safe)
                    s_logits = s_logits.float(); newval = newval.float()
                    vdist = vdist.float() if vdist is not None else None
                pdist = Categorical(logits=s_logits)            # full student action dist (also for teacher-KL)
                newlogp = pdist.log_prob(b_actions[mb]); entropy = pdist.entropy()
                ratio = (newlogp - b_logp[mb]).exp()
                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                    approx_kls.append((((ratio - 1.0) - (newlogp - b_logp[mb])).mean()).item())  # Schulman approx-KL
                adv = b_adv[mb]
                if args.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                if net.value_categorical:                       # cross-entropy to the two-hot-projected return
                    vtgt = net.value_target_dist(b_returns[mb])
                    v_loss = -(vtgt * torch.log_softmax(vdist, dim=-1)).sum(-1).mean()
                elif args.vf_clip is not None:                  # PPO value clip: trust region around the old value
                    vc = b_vals[mb] + torch.clamp(newval - b_vals[mb], -args.vf_clip, args.vf_clip)
                    v_loss = 0.5 * torch.max((newval - b_returns[mb]) ** 2, (vc - b_returns[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newval - b_returns[mb]) ** 2).mean()
                ent_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss
                if t_cache is not None:                         # B6: anti-regression distillation to last-best
                    t_logp = torch.log_softmax(t_cache["logits"][mb], dim=-1)
                    s_logp = torch.log_softmax(s_logits, dim=-1)
                    kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()   # KL(teacher || student)
                    if net.value_categorical:                   # distil the value DISTRIBUTION (CE)
                        tvp = torch.softmax(t_cache["vdist"][mb], dim=-1)
                        vdistill = -(tvp * torch.log_softmax(vdist, dim=-1)).sum(-1).mean()
                    else:                                       # scalar head -> MSE distill (point estimate)
                        vdistill = 0.5 * ((newval - t_cache["val"][mb]) ** 2).mean()
                    loss = loss + args.teacher_kl_coef * kl + args.teacher_value_coef * vdistill
                opt.zero_grad()
                loss.backward()
                if ddp:                                  # average gradients across ranks -> identical step
                    for p in net.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad)
                            p.grad /= world
                nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                opt.step()
                last = {"pg": pg_loss.item(), "v": v_loss.item(), "ent": ent_loss.item()}
            if args.target_kl is not None and approx_kls:          # KL early-stop: bound the trust region
                nmb = max(1, args.batch_size // args.minibatch_size)
                ek = float(np.mean(approx_kls[-nmb:]))             # this epoch's mean approx-KL
                if ddp:                                            # all ranks must stop together
                    t = torch.tensor(ek, device=device); dist.all_reduce(t); ek = (t / world).item()
                if ek > args.target_kl:
                    break
        return clipfracs, approx_kls, last

    bufs = [make_buffers() for _ in range(2 if args.async_collect else 1)]
    behavior_net = None
    if args.async_collect:
        # frozen behavior-policy clone for the background collector (no race with the optimizer)
        behavior_net = build_token_net(ct, net_config).to(collector_device)
        behavior_net.load_state_dict(net.state_dict())
        behavior_net.eval()
        if is_main:
            print(f"[async] overlapped collection (frozen snapshot, 1-update stale); "
                  f"collector fwd on {collector_device}, learner on {device}", flush=True)

    global_step = resume_step
    start = time.time()
    next_obs_np, _ = envs.reset()
    next_obs = to_tensors(next_obs_np, device)
    next_done = torch.zeros(args.num_envs, device=device)

    snapshot_pool = collections.deque(maxlen=args.pool_size)
    selfplay_on = False
    last_snapshot = 0
    ep_returns = collections.deque(maxlen=200)  # terminal rewards (~win signal)

    # start self-play immediately (e.g. resuming a model that already beats random)
    if args.selfplay_start <= 0:
        selfplay_on = True
        snapshot_pool.append({k: v.detach().clone().cpu() for k, v in net.state_dict().items()})
        envs.set_opponent(snapshot_pool[-1])
        refresh_teacher()
        print("[selfplay] enabled at start (selfplay_start<=0)", flush=True)

    if resume_step > 0 and args.selfplay_start > 0 and resume_step >= args.selfplay_start and not selfplay_on:
        selfplay_on = True                                   # resumed past warmup -> re-seed opponent pool
        snapshot_pool.append({k: v.detach().clone().cpu() for k, v in net.state_dict().items()})
        envs.set_opponent(snapshot_pool[-1])
        refresh_teacher()
        last_snapshot = resume_step
        if is_main:
            print(f"[resume] self-play re-enabled at step {resume_step}", flush=True)

    def selfplay_schedule():
        nonlocal selfplay_on, last_snapshot
        cur_sd = lambda: {k: v.detach().clone().cpu() for k, v in net.state_dict().items()}
        if not selfplay_on and global_step >= args.selfplay_start:
            selfplay_on = True
            snapshot_pool.append(cur_sd())
            envs.set_opponent(snapshot_pool[-1])
            refresh_teacher()
            last_snapshot = global_step
            print(f"[selfplay] enabled at step {global_step}", flush=True)
        if selfplay_on and global_step - last_snapshot >= args.snapshot_every:
            cur = cur_sd()
            snapshot_pool.append(cur)
            # C10 pure-live: opponent = the LATEST current net; else a random pool member (anti-cycling).
            pick = cur if args.pure_live else snapshot_pool[np.random.randint(len(snapshot_pool))]
            envs.set_opponent(pick)
            refresh_teacher()    # time-based teacher refresh; --promote-winrate (gated) = future in-proc eval
            last_snapshot = global_step

    t_collect = t_update = 0.0
    num_iters = args.total_timesteps // (args.batch_size * world)   # total_timesteps is the GLOBAL budget
    decay_iters = (args.decay_steps // (args.batch_size * world)) if args.decay_steps \
        else max(1, int(0.7 * num_iters))                           # default: decay over 70% of the budget
    if args.warmup_frac > 0:                                        # deprecated %-mode
        warmup_iters = int(args.warmup_frac * decay_iters)
    else:                                                           # FLAT warmup (budget-invariant)
        warmup_iters = min(args.warmup_steps // (args.batch_size * world), max(1, num_iters // 5))
    start_it = (global_step // (args.batch_size * world)) + 1        # resume continues the LR schedule here
    cur = 0
    pending = None
    if args.async_collect:                                    # prime the pipeline with rollout 0
        pending = collect_rollout(behavior_net, next_obs, next_done, bufs[0], False, collector_device)
        global_step += pending["nsteps"] * world; ep_returns.extend(pending["eprs"]); t_collect = pending["tc"]
        next_obs, next_done = pending["end_obs"], pending["end_done"]

    for it in range(start_it, num_iters + 1):
        if args.anneal_lr:
            opt.param_groups[0]["lr"] = lr_at(it, decay_iters, args.lr,
                                              args.lr_schedule, warmup_iters, args.lr_min_ratio)

        if args.async_collect:
            # snapshot the current (pre-update) net as the behavior policy for the NEXT rollout,
            # then collect it on a background thread while the learner updates on the CURRENT rollout.
            behavior_net.load_state_dict(net.state_dict())
            nxt = 1 - cur
            start_obs, start_done = pending["end_obs"], pending["end_done"]
            holder = {}
            th = threading.Thread(target=lambda: holder.update(
                out=collect_rollout(behavior_net, start_obs, start_done, bufs[nxt], False, collector_device)))
            th.start()
            _tu0 = time.time()
            advantages, returns = compute_gae(bufs[cur], pending["boot_val"], pending["end_done"])
            clipfracs, approx_kls, last = update_net(bufs[cur], advantages, returns, teacher=teacher_net)
            t_update = time.time() - _tu0
            th.join()
            pending = holder["out"]
            global_step += pending["nsteps"] * world; ep_returns.extend(pending["eprs"]); t_collect = pending["tc"]
            next_obs, next_done = pending["end_obs"], pending["end_done"]
            selfplay_schedule()
            cur = nxt
        else:
            pending = collect_rollout(net, next_obs, next_done, bufs[0], _PROF, device)
            global_step += pending["nsteps"] * world; ep_returns.extend(pending["eprs"]); t_collect = pending["tc"]
            next_obs, next_done = pending["end_obs"], pending["end_done"]
            selfplay_schedule()
            if _PROF and device.type == "cuda":
                torch.cuda.synchronize()
            _tu0 = time.time()
            advantages, returns = compute_gae(bufs[0], pending["boot_val"], pending["end_done"])
            clipfracs, approx_kls, last = update_net(bufs[0], advantages, returns, teacher=teacher_net)
            if _PROF and device.type == "cuda":
                torch.cuda.synchronize()
            t_update = time.time() - _tu0

        # ---- logging (rank 0 only) ----
        if is_main and it % args.log_every == 0:
            # sps = THIS iteration's rate; avg = process mean since resume (see train_selfplay)
            sps = int(args.batch_size * world / max(t_collect + t_update, 1e-9))
            avg = int((global_step - resume_step) / max(time.time() - start, 1e-9))
            wr = float(np.mean([rr > 0 for rr in ep_returns])) if ep_returns else float("nan")
            mret = float(np.mean(ep_returns)) if ep_returns else float("nan")
            prof = (f" collect={t_collect:.1f}s update={t_update:.1f}s "
                    f"(c-sps={int(args.batch_size * world / max(t_collect,1e-9))})") if _PROF else ""
            print(f"step={global_step} it={it}/{num_iters} sps={sps} avg={avg} "
                  f"winrate={wr:.3f} ep_ret={mret:+.3f} "
                  f"pg={last['pg']:.3f} v={last['v']:.3f} ent={last['ent']:.3f} "
                  f"clip={np.mean(clipfracs):.3f} kl={np.mean(approx_kls):.4f} selfplay={selfplay_on}{prof}", flush=True)

        # ---- checkpoint (rank 0; nets identical across ranks) ----
        if is_main and (global_step % args.save_every < args.batch_size * world or it == num_iters):
            path = os.path.join(args.out, f"ckpt_{global_step}.pt")
            torch.save({"net": net.state_dict(), "args": vars(args),     # net-only: eval / pick / export
                        "net_config": net_config, "global_step": global_step}, path)
            torch.save({"net": net.state_dict(), "args": vars(args), "opt": opt.state_dict(),
                        "net_config": net_config, "global_step": global_step},   # latest = FULL resume state
                       os.path.join(args.out, "latest.pt"))
            print(f"[ckpt] saved {path}", flush=True)

    envs.close()
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        print("done.", flush=True)


if __name__ == "__main__":
    main()
