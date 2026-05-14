# SDFT Algorithm

Self-Distillation Fine-Tuning (Shenfeld et al. 2026) trains a model to follow new instructions while minimising forgetting of its prior policy.

## Roles

| | Sees | Updated by |
|---|---|---|
| **Student** π_θ | Prompt x only | AdamW gradient |
| **Teacher** π_φ | Prompt x + demonstration c | EMA of student weights |

At t=0 both share the same weights. The teacher drifts more slowly than the student, acting as a smoothed proxy for "what a well-instructed model should say."

## Loss

    L_SDFT = E_{y ~ π_θ(·|x)} [ log π_θ(y|x) − log π_φ(y|x,c) ]

This is a *reverse KL* from the student distribution to the teacher distribution, estimated by a single Monte-Carlo sample y drawn from the student.

The gradient flows only through the student log-probs; teacher log-probs are treated as a constant target (stop_gradient).

## Data flow (one training step)

```
                     x_ids
                       │
            ┌──────────▼──────────┐
            │   sample_rollout    │   y ~ π_student(·|x)   [no grad]
            └──────────┬──────────┘
                       │ y_ids
           ┌───────────┼──────────────────┐
           │           │                  │
   x ++ y  │       x ++ c ++ y            │
     ┌─────▼──────┐  ┌──────────────┐     │
     │  student   │  │   teacher    │     │
     │ (gradient) │  │ (stop_grad)  │     │
     └─────┬──────┘  └──────┬───────┘     │
           │                │             │
      log_p_student    log_p_teacher      │
           │                │             │
           └──────┬─────────┘             │
                  │                       │
        mean(log_p_s - log_p_t)           │
                  │                       │
                loss                      │
                  │                       │
              backward                   │
                  │                       │
            AdamW.update ◄────────────────┘
                  │
         EMATeacher.update
                  │
         teacher ← α·teacher + (1−α)·student
```

## EMA teacher update

    φ_t = α · φ_{t-1} + (1 − α) · θ_t

Typical α = 0.999 (full runs) or 0.99 (smoke tests). Small α makes the teacher track the student quickly; large α preserves the base policy structure longer. The paper uses α = 0.999 for 3B-scale experiments.

## Why reverse-KL?

Forward KL (KL[p_teacher || p_student]) would require importance-weighted samples from the teacher and is variance-heavy. Reverse KL (KL[p_student || p_teacher]) is estimated cheaply with a single student rollout. It has mode-seeking behaviour, which encourages the student to cover the high-probability modes of the teacher's distribution — appropriate for instruction following where we want the student to produce plausible responses, not necessarily all of them.

## Gradient signal as affectedness proxy

The per-parameter gradient magnitude of the SDFT loss under a given training sample is a direct measure of how much that parameter's current value conflicts with what the task requires. This is the signal used in Phase 1 (Borda probe) of the research experiment — see `hypothesis.md` in the research repo for details.
