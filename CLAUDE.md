# HRT / Algo Arena workspace

The full operating brief is in `docs/OPERATING_SKILL.md`. Read it at the start of any competition work. Summary of the rules that matter most:

- **Two modes, never mixed.** Mode A = algorithm engineering (puzzles, CP, optimisation). Mode B = quantitative trading (Algo Arena etc.). Don't carry assumptions across.
- **Understand before building.** Extract the exact rules, interface, scoring and deadline from the provided materials. Label everything as *Confirmed*, *Hypothesis* or *Unknown*. Never invent rules.
- **Scoring decides the objective.** Write the score down as a formula before optimising.
- **Baseline early, fallback always.** A valid submission must exist before advanced work. Keep the last known-good version tagged.
- **Evidence over intuition.** Every claimed improvement needs a controlled, recorded experiment. In trading: out-of-sample / walk-forward, costs included, no look-ahead, no repeated tuning on one validation set.
- **Historical research must be honest.** Never claim a strategy or team won an event unless a source establishes that exact link. Rate source reliability.
- **Independent review.** Advisor (strategy challenge) and auditor (find failures) at the checkpoints in §4–5 of the brief. If a real /advisor is not available, say so and run a labelled internal adversarial review instead.
- **Master state.** Keep `ALGO_ARENA_MASTER_STATE.md` current after every new material, experiment, review or code change.
- Never claim code was run or benchmarked unless it was.
