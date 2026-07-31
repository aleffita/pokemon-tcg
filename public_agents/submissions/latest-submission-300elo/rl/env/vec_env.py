"""Subprocess vector env for cabt.

The native engine holds ONE global battle pointer, so each env must live in its
own process. This runs ``num_envs`` workers, each owning a single ``CabtEnv``,
with auto-reset on episode end and a channel to broadcast opponent-policy weights
for self-play.

Two opponent execution modes (``opponent_mode``):

* ``"local"`` (default): the opponent net runs INSIDE each worker on CPU. A central
  server was once tried for the MLP and measured ~2-3x SLOWER -- one process
  serializing IPC round-trips beat per-worker local forwards because the MLP forward
  is sub-ms (round-trip dominates).

* ``"server"``: the opponent ENCODES locally (cheap) but routes the forward to a
  batched inference server -- a THREAD in the main process holding the snapshot net
  on the GPU. While the main thread blocks in ``step`` waiting on workers, the server
  thread batches the workers' concurrent forward requests into ONE GPU forward. This
  WINS for the v2 transformer (CPU forward is expensive, so per-worker forwards thrash
  the cores), the opposite regime to the cheap MLP. No second CUDA context (shares the
  main process's), no weight pickling, and workers hold no net.

  The encoded obs (fixed shapes) crosses to the server via SHARED MEMORY, not the pipe:
  each worker writes its obs into row ``i`` of a shared batch buffer and signals the
  server with just its index (a few bytes); the server reads the ready rows as a
  zero-copy numpy view, forwards on GPU, writes logits into a shared response buffer,
  and acks. This removes the ~40 KB pickle/unpickle per request that otherwise caps the
  single server thread.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from multiprocessing import resource_tracker, shared_memory
from multiprocessing.connection import wait

import numpy as np


def _policy_opponent_factory(net_config):
    """Worker-local opponent playing a frozen snapshot of the v2 net (raw, not jit-frozen --
    the token transformer doesn't trace cleanly). Returns (opponent_fn, set_weights)."""
    import torch
    from rl.encoder.card_features import get_card_table
    torch.set_num_threads(1)

    # token transformer: the env threads the opponent's true deck + shared tracker; no jit (trace is fragile)
    from rl.encoder.encoding import SUBMIT_ACTION
    from rl.encoder.encoding import TokenEncoder
    from rl.policy import build_token_net
    enc = TokenEncoder(get_card_table())
    state = {"net": None, "nenc": None}                  # nenc: lazily-built NativeEncoder (JSON-free opponent)
    import os
    _opp_first = os.environ.get("V2_OPP_FIRST") == "1"   # experiment: opponent always plays the FIRST emitted option(s), not uniform

    def set_weights(sd):
        if sd is None:
            state["net"] = None
            return
        sd = {k: torch.as_tensor(v) for k, v in sd.items()}   # numpy (from the pipe) -> tensors
        net = build_token_net(enc.cards, net_config)
        net.load_state_dict(sd); net.eval()
        state["net"] = net                           # raw net (transformer doesn't jit-freeze cleanly)

    @torch.no_grad()
    def opponent_fn(raw_obs, rng, deck=None, tracker=None, ability_slots=None, binary=None):
        net = state["net"]
        sel = raw_obs["select"]
        if net is None:                              # warmup / no snapshot
            n, k = len(sel["option"]), sel["maxCount"]
            if not n:
                return []
            if _opp_first:                           # V2_OPP_FIRST=1: deterministic first-move bot (not uniform)
                return list(range(min(k, n)))
            return rng.sample(range(n), min(k, n))   # default: random legal
        if binary is not None and state["nenc"] is None:      # native path: build the shared encoder once
            from rl.native_enc import NativeEncoder
            state["nenc"] = NativeEncoder(enc)
        if binary is None and net_config.get("would_ko") and deck is not None:   # JSON path only (native = no would_ko)
            try:                                              # match the learner: annotate opp attacks once/decision
                from rl import search_agent as _SA
                _SA.annotate_would_ko(raw_obs, deck, enc)
            except Exception:
                pass
        picked: list[int] = []
        while True:
            if binary is not None:                            # JSON-free: encode straight from the binary buffer
                enc_obs = state["nenc"].encode(binary, sel, deck, tracker, ability_slots, picked)
            else:
                enc_obs = enc.encode(raw_obs, set(picked), self_deck=deck, tracker=tracker,
                                     ability_slots=ability_slots)
            o = {k: torch.as_tensor(v[None], dtype=(torch.long if k in enc.int_keys else torch.float32))
                 for k, v in enc_obs.items()}
            logits, _ = net.logits_value(o)
            a = int(logits.argmax(-1).item())
            if a == SUBMIT_ACTION:
                break
            picked.append(a)
            if len(picked) >= sel["maxCount"]:
                break
        return sorted(set(picked))

    return opponent_fn, set_weights


def _attach_shm(name):
    """Attach to a main-owned shared-memory block. Unregister from THIS process's
    resource_tracker so worker exit doesn't try to unlink a block the main owns."""
    shm = shared_memory.SharedMemory(name=name)
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass
    return shm


def _server_opponent_factory(net_config, opp_remote, srv):
    """Opponent that ENCODES locally (cheap) and routes the FORWARD to the main-process
    server: writes its obs into shared-memory row ``srv['idx']``, signals the server with
    its index, and reads logits back from the shared response row. No per-worker net.
    ``set_mode(bool)`` toggles server use (False = random legal, the warmup curriculum)."""
    import torch
    from rl.encoder.card_features import get_card_table
    torch.set_num_threads(1)

    from rl.encoder.encoding import SUBMIT_ACTION
    from rl.encoder.encoding import TokenEncoder
    from rl.policy import build_token_net
    enc = TokenEncoder(get_card_table())

    idx, n = srv["idx"], srv["n"]
    shms = {k: _attach_shm(name) for k, name in srv["names"].items()}
    bufs = {k: np.ndarray((n, *shp), dtype=dt, buffer=shms[k].buf)
            for k, (shp, dt) in srv["spec"].items()}
    resp_shm = _attach_shm(srv["resp"])
    resp = np.ndarray((n, srv["resp_w"]), dtype=np.float32, buffer=resp_shm.buf)
    # CRITICAL: keep the SharedMemory objects alive for the closure's lifetime. The numpy
    # views (bufs/resp) hold only the mmap memoryview; if the SharedMemory objects are GC'd
    # their __del__ closes the mmap -> the views point to freed memory -> C-level SEGFAULT
    # on the next access (silent: no catchable Python exception). Stash them in `state`.
    state = {"use_server": False, "_keep": (shms, resp_shm), "nenc": None}

    def set_mode(flag):
        state["use_server"] = bool(flag)

    def _random(sel, rng):
        m, k = len(sel["option"]), sel["maxCount"]
        return rng.sample(range(m), min(k, m)) if m else []

    def opponent_fn(raw_obs, rng, deck=None, tracker=None, ability_slots=None, binary=None):
        sel = raw_obs["select"]
        if not state["use_server"]:                  # warmup: random legal, no round-trip
            return _random(sel, rng)
        if binary is not None and state["nenc"] is None:      # native path: build the shared encoder once
            from rl.native_enc import NativeEncoder
            state["nenc"] = NativeEncoder(enc)
        if binary is None and net_config.get("would_ko") and deck is not None:   # JSON path only (native = no would_ko)
            try:                                              # match the learner: annotate opp attacks once/decision
                from rl import search_agent as _SA
                _SA.annotate_would_ko(raw_obs, deck, enc)
            except Exception:
                pass
        picked: list[int] = []
        while True:
            if binary is not None:                            # JSON-free: encode straight from the binary buffer
                enc_obs = state["nenc"].encode(binary, sel, deck, tracker, ability_slots, picked)
            else:
                enc_obs = enc.encode(raw_obs, set(picked), self_deck=deck, tracker=tracker,
                                     ability_slots=ability_slots)
            for k, v in enc_obs.items():
                bufs[k][idx] = v                     # write my row in shared memory
            opp_remote.send(idx)                     # tiny signal (few bytes)
            if not opp_remote.poll(10.0):            # server unresponsive -> degrade, never hang
                return _random(sel, rng)
            ok = opp_remote.recv()                   # ack (True), or None on server error
            if ok is None:
                return _random(sel, rng)
            a = int(np.argmax(resp[idx]))            # logits already action-masked by the net
            if a == SUBMIT_ACTION:
                break
            picked.append(a)
            if len(picked) >= sel["maxCount"]:
                break
        return sorted(set(picked))

    return opponent_fn, set_mode


def _scripted_opponent_factory(main_py_path):
    """FIXED scripted opponent from a Kaggle-style main.py (`def agent(obs_dict) -> list[int]`).
    Exec'd ONCE per worker; `cg` must be importable (we vendor cg as `cg`). The module reads
    its deck.csv at import, so we exec with CWD = the main.py's dir. set_opp is a no-op: the
    opponent is fixed (no self-play snapshot ever overrides it)."""
    import os, importlib
    importlib.import_module("cg.api")          # cache `cg` BEFORE chdir so B's `from cg.api` is CWD-independent
    d = os.path.dirname(os.path.abspath(main_py_path))
    ns = {"__name__": "_scripted_opp", "__file__": os.path.abspath(main_py_path)}
    cwd = os.getcwd()
    try:
        os.chdir(d)                            # B reads its deck.csv relative to its own dir at import
        with open(main_py_path, encoding="utf-8") as f:
            exec(compile(f.read(), main_py_path, "exec"), ns)
    finally:
        os.chdir(cwd)
    agent = ns["agent"]

    def opponent_fn(raw_obs, rng, deck=None, tracker=None, ability_slots=None):
        try:
            out = agent(raw_obs)
            return list(out) if out else []
        except Exception:                                    # never let a buggy opponent kill the worker
            sel = raw_obs.get("select") or {}
            m, k = len(sel.get("option", [])), sel.get("maxCount", 1)
            return rng.sample(range(m), min(k, m)) if m else []

    def set_opp(_):                                          # scripted opponent is fixed
        pass
    return opponent_fn, set_opp


def _worker(remote, parent_remote, env_kwargs, net_config, seed, srv=None, obs_shm=None):
    parent_remote.close()
    import logging; logging.disable(logging.CRITICAL)
    _worker_main(remote, env_kwargs, net_config, seed, srv, obs_shm)


def _worker_main(remote, env_kwargs, net_config, seed, srv=None, obs_shm=None):
    from rl.env.env import CabtEnv, prize_diff_shaping

    env_kwargs = dict(env_kwargs)
    shaping = env_kwargs.pop("shaping", None)
    scripted = env_kwargs.pop("scripted_opponent", None)    # path to a fixed Kaggle main.py opponent
    reward_shaping = prize_diff_shaping(0.1) if shaping == "prize_diff" else None

    if scripted is not None:                       # FIXED external scripted opponent (no self-play)
        opponent_fn, set_opp = _scripted_opponent_factory(scripted)
    elif srv is not None:                          # server mode: forward routed to main proc
        opponent_fn, set_opp = _server_opponent_factory(net_config, srv["opp"], srv)
    else:                                          # local mode: net runs in-worker
        opponent_fn, set_opp = _policy_opponent_factory(net_config)
    from rl.encoder.encoding import TokenEncoder
    from rl.encoder.card_features import get_card_table
    encoder = TokenEncoder(get_card_table())
    env = CabtEnv(seed=seed, opponent_fn=opponent_fn, reward_shaping=reward_shaping,
                  encoder=encoder, **env_kwargs)

    # Shared-memory obs: write the encoded obs straight into this worker's row of a shared batch
    # buffer instead of pickling it over the Pipe (the main then reads the whole batch zero-copy).
    # The Pipe still carries the small (r, done, info) tuple -- that recv also barriers the main
    # until every worker has finished writing its row, so the batch is fully populated on read.
    _write_obs = None
    if obs_shm is not None:
        _idx, _n = obs_shm["idx"], obs_shm["n"]
        _oshm, _obufs = [], {}
        for k, (shp, dt) in obs_shm["spec"].items():
            s = _attach_shm(obs_shm["names"][k])          # unregister from THIS proc's resource_tracker
            _oshm.append(s)
            _obufs[k] = np.ndarray((_n, *shp), dtype=dt, buffer=s.buf)
        _zero_obs = {k: np.zeros(shp, dtype=dt) for k, (shp, dt) in obs_shm["spec"].items()}

        def _write_obs(o, _obufs=_obufs, _idx=_idx, _zero=_zero_obs):
            if o is None:                                 # recovery couldn't produce an obs -> write
                o = _zero                                 # zeros (a neutral boundary) so we never
            for k in _obufs:                              # crash on o[k]/hang the main on recv
                _obufs[k][_idx] = o[k]

    try:
        last_obs = None                                   # last successfully-produced obs (shape donor)
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                obs, info = env.reset()
                last_obs = obs
                if _write_obs is not None:
                    _write_obs(obs); remote.send(info)
                else:
                    remote.send((obs, info))
            elif cmd == "step":
                try:
                    obs, r, term, trunc, info = env.step(data)
                    done = term or trunc
                    if done:
                        info = {**info, "truncated": trunc}
                        if term:                       # only a real game end has a terminal reward;
                            info["terminal_reward"] = r  # truncation (max_steps) is not terminal
                        obs, _ = env.reset()  # auto-reset
                except Exception:
                    # a bad game/engine state must NOT kill the worker -> it would cascade
                    # (ConnectionReset in the main) and fail the whole job. Recover: reset and
                    # report a neutral episode boundary; one lost transition >> a dead run.
                    try:
                        obs, _ = env.reset()
                    except Exception:
                        try:
                            obs = env._encode()
                        except Exception:
                            obs = None
                    r, done, info = 0.0, True, {"recovered": True}
                if obs is None:                           # doubly-failed recovery: zero obs (neutral,
                    obs = {k: np.zeros_like(v) for k, v in last_obs.items()}  # all-masked) -- the non-shm
                last_obs = obs                            # _stack in the main must NEVER see None
                if _write_obs is not None:
                    _write_obs(obs); remote.send((r, done, info))
                else:
                    remote.send((obs, r, done, info))
            elif cmd == "set_opponent":            # local: data=state_dict|None ; server: data=use_server bool
                set_opp(data)
                remote.send(True)
            elif cmd == "close":
                env.close()
                remote.send(True)
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        env.close()


def _ref_spec(net_config, env_kwargs):
    """Per-key encoded-obs (shape, dtype) from a throwaway CabtEnv reset -- shapes are
    fixed (the encoder pads to constants), so one sample defines the shared buffers."""
    from rl.env.env import CabtEnv
    from rl.encoder.card_features import get_card_table
    ek = dict(env_kwargs); ek.pop("shaping", None); ek.pop("scripted_opponent", None)
    from rl.encoder.encoding import TokenEncoder
    from rl.policy import build_token_net
    enc = TokenEncoder(get_card_table())
    env = CabtEnv(seed=0, encoder=enc, **ek)
    try:
        obs, _ = env.reset()
    finally:
        env.close()
    return {k: (tuple(v.shape), v.dtype) for k, v in obs.items()}


class SubprocVecEnv:
    def __init__(self, num_envs: int, env_kwargs: dict, net_config: dict,
                 base_seed: int = 0, start_method: str | None = None,
                 server_device: str = "cpu", opponent_mode: str = "local",
                 obs_shm: bool = False):
        self.num_envs = num_envs
        self.opponent_mode = opponent_mode
        self.net_config = net_config
        ctx = mp.get_context(start_method or ("spawn"))
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])

        srvs = [None] * num_envs
        self._shm = []

        # Learner-obs shared-memory batch: workers write their encoded rollout obs into a shared
        # [num_envs, *shape] buffer (per key); step()/reset() then return zero-copy views of it
        # instead of unpickling + np.stack-ing N per-worker obs dicts. Orthogonal to server mode
        # (whose _obs_bufs route the OPPONENT's decision obs to the inference server).
        self._obs_shm = bool(obs_shm)
        obs_shms = [None] * num_envs
        if self._obs_shm:
            lspec = _ref_spec(net_config, env_kwargs)
            lnames = {}
            self._learner_obs_bufs = {}
            for k, (shp, dt) in lspec.items():
                nbytes = max(num_envs * int(np.prod(shp)) * np.dtype(dt).itemsize, 1)
                shm = shared_memory.SharedMemory(create=True, size=nbytes)
                self._shm.append(shm); lnames[k] = shm.name
                self._learner_obs_bufs[k] = np.ndarray((num_envs, *shp), dtype=dt, buffer=shm.buf)
            for i in range(num_envs):
                obs_shms[i] = {"idx": i, "n": num_envs, "names": lnames, "spec": lspec}
        if opponent_mode == "server":
            spec = _ref_spec(net_config, env_kwargs)               # {k: (shape, dtype)}
            resp_w = int(np.prod(spec["action_mask"][0]))          # N_ACTIONS
            names = {}
            self._obs_bufs = {}
            for k, (shp, dt) in spec.items():
                nbytes = max(num_envs * int(np.prod(shp)) * np.dtype(dt).itemsize, 1)
                shm = shared_memory.SharedMemory(create=True, size=nbytes)
                self._shm.append(shm); names[k] = shm.name
                self._obs_bufs[k] = np.ndarray((num_envs, *shp), dtype=dt, buffer=shm.buf)
            resp_shm = shared_memory.SharedMemory(create=True, size=num_envs * resp_w * 4)
            self._shm.append(resp_shm)
            self._resp = np.ndarray((num_envs, resp_w), dtype=np.float32, buffer=resp_shm.buf)

            opp_pairs = [ctx.Pipe() for _ in range(num_envs)]
            self.opp_conns = [a for a, _ in opp_pairs]             # main-side ends (server reads)
            for i in range(num_envs):
                srvs[i] = {"idx": i, "n": num_envs, "names": names, "spec": spec,
                           "resp": resp_shm.name, "resp_w": resp_w, "opp": opp_pairs[i][1]}
        else:
            self.opp_conns = None

        self.procs = []
        for i, (wr, r) in enumerate(zip(self.work_remotes, self.remotes)):
            p = ctx.Process(target=_worker,
                            args=(wr, r, env_kwargs, net_config, base_seed + i, srvs[i], obs_shms[i]),
                            daemon=True)
            p.start()
            self.procs.append(p)
        for wr in self.work_remotes:
            wr.close()
        if opponent_mode == "server":
            for i in range(num_envs):
                srvs[i]["opp"].close()                             # workers hold their own copies
            self._start_server(net_config, server_device)

    # ---- batched inference server (thread in THIS process) --------------------
    def _start_server(self, net_config, device):
        import torch
        from rl.encoder.card_features import get_card_table
        dev = torch.device(device)
        from rl.encoder.encoding import TokenEncoder
        from rl.policy import build_token_net
        enc = TokenEncoder(get_card_table())
        net = build_token_net(enc.cards, net_config)
        self._srv_net = net.to(dev).eval()
        self._srv_dev = dev
        self._srv_lock = threading.Lock()
        self._srv_stop = threading.Event()
        import os
        # Dynamic-batching window. Default OFF: each opponent decision is on a worker's
        # critical path, so waiting to coalesce bigger batches adds latency that hurts more
        # than it helps (measured: 0ms best, monotonically worse to 4ms). Tunable for other regimes.
        self._srv_coalesce = float(os.environ.get("SRV_COALESCE_MS", "0")) / 1000.0
        self._srv_thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._srv_thread.start()

    def _serve_loop(self):
        import torch
        conns = list(self.opp_conns)
        keys = list(self._obs_bufs.keys())
        coalesce = self._srv_coalesce                          # dynamic-batching window (s)
        while not self._srv_stop.is_set():
            ready = wait(conns, timeout=0.25)
            if not ready:
                continue
            # Dynamic batching: wait() returns the instant ONE worker is ready, but workers
            # are staggered by between-decision engine work -> tiny GPU batches. Briefly pull
            # in stragglers so one GPU forward serves many workers (fewer host<->GPU syncs).
            ready = list(ready)
            if coalesce > 0:
                remaining = [c for c in conns if c not in ready]
                if remaining:
                    ready += list(wait(remaining, timeout=coalesce))
            pend = []                                              # (conn, worker_idx)
            for c in ready:
                try:
                    pend.append((c, c.recv()))
                except EOFError:
                    conns.remove(c)
            if not pend:
                continue
            rows = [idx for _, idx in pend]
            try:
                batch = {k: torch.as_tensor(self._obs_bufs[k][rows], device=self._srv_dev)
                         for k in keys}
                with torch.no_grad(), self._srv_lock:
                    logits = self._srv_net.logits_value(batch)[0]
                self._resp[rows] = logits.cpu().numpy()            # write shared response rows
                for c, _ in pend:
                    c.send(True)                                   # ack: response ready
            except Exception:
                for c, _ in pend:                                  # never deadlock the workers
                    try:
                        c.send(None)
                    except Exception:
                        pass

    def _stack(self, obs_list):
        return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        if self._obs_shm:                                  # workers wrote their rows; recv barriers them
            infos = [r.recv() for r in self.remotes]
            return dict(self._learner_obs_bufs), list(infos)
        obs, infos = zip(*[r.recv() for r in self.remotes])
        return self._stack(obs), list(infos)

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", int(a)))
        if self._obs_shm:                                  # obs already in the shared batch buffer
            rews, dones, infos = zip(*[r.recv() for r in self.remotes])
            return (dict(self._learner_obs_bufs),
                    np.asarray(rews, dtype=np.float32),
                    np.asarray(dones, dtype=np.bool_),
                    list(infos))
        results = [r.recv() for r in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return (self._stack(obs),
                np.asarray(rews, dtype=np.float32),
                np.asarray(dones, dtype=np.bool_),
                list(infos))

    def pin_obs_buffers(self):
        """Page-lock (cudaHostRegister) the learner-obs shm buffers so per-step H2D copies to GPU use
        DMA instead of a pageable staging copy. Best-effort: no-op if not obs_shm / no CUDA / the
        registration is unavailable. Call ONCE after CUDA is initialized (e.g. after net.to(cuda))."""
        self._pinned = []
        if not self._obs_shm:
            return
        try:
            import torch
            if not torch.cuda.is_available():
                return
            cudart = torch.cuda.cudart()
            for arr in self._learner_obs_bufs.values():
                cudart.cudaHostRegister(arr.ctypes.data, arr.nbytes, 0)   # 0 = cudaHostRegisterDefault
                self._pinned.append(arr.ctypes.data)
            import sys; print(f"[pin-obs] page-locked {len(self._pinned)} shm obs buffers", file=sys.stderr, flush=True)
        except Exception as e:
            import sys; print(f"[pin-obs] skipped ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
            self._pinned = []

    def set_opponent(self, state_dict):
        """Set the snapshot opponent (or None = random).

        server mode: load the snapshot into the server net (one place) and tell workers
        to route forwards to it (None -> workers fall back to random legal). local mode:
        broadcast the weights to each worker as NUMPY -- sending torch tensors over mp
        uses fd-based storage sharing, and many workers x many tensors exhausts the open
        file limit ('Too many open files') at the broadcast; numpy pickles by value."""
        if self.opponent_mode == "server":
            if state_dict is not None:
                import torch
                sd = {k: torch.as_tensor(v) for k, v in state_dict.items()}
                with self._srv_lock:
                    self._srv_net.load_state_dict(sd)
                    self._srv_net.eval()
            flag = state_dict is not None
            for r in self.remotes:
                r.send(("set_opponent", flag))
            return [r.recv() for r in self.remotes]

        payload = (None if state_dict is None
                   else {k: v.detach().cpu().numpy() for k, v in state_dict.items()})
        for r in self.remotes:
            r.send(("set_opponent", payload))
        return [r.recv() for r in self.remotes]

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None)); r.recv()
            except Exception:
                pass
        if self.opponent_mode == "server":
            self._srv_stop.set()
            if self._srv_thread.is_alive():
                self._srv_thread.join(timeout=5)
        for _ptr in getattr(self, "_pinned", []):          # unregister page-locked obs bufs first
            try:
                import torch; torch.cuda.cudart().cudaHostUnregister(_ptr)
            except Exception:
                pass
        for shm in self._shm:                              # server _obs_bufs/_resp + learner obs bufs
            try:
                shm.close(); shm.unlink()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
