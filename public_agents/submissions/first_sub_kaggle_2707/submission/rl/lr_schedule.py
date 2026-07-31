import math

def lr_at(it, decay_iters, base_lr, schedule, warmup_iters, min_ratio):
    """LR for iteration `it` (1-based): linear warmup, then linear/cosine decay over `decay_iters` to
    base_lr*min_ratio, then FLAT at the floor (Orbit-style: decay window decoupled from total budget;
    `decay_step` is clamped so past the window the LR holds at the floor). Pure function of `it` ->
    resume-safe (start_it continues the same curve). Shared by train + train_selfplay."""
    if warmup_iters > 0 and it <= warmup_iters:
        return base_lr * it / warmup_iters
    prog = min(max((it - warmup_iters) / max(1, decay_iters), 0.0), 1.0)   # clamp -> FLAT at floor after window
    decay = 0.5 * (1.0 + math.cos(math.pi * prog)) if schedule == "cosine" else (1.0 - prog)
    return base_lr * (min_ratio + (1.0 - min_ratio) * decay)