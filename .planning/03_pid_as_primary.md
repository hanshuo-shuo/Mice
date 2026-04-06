# PID Controllers as the Primary Analytical Frame

## The observation
The PID baseline produced surprisingly good trajectories. Two parameters — `risk_tolerance` and `partner_trust` — generate the full range of behaviors we care about: wall-following, waiting, predator avoidance, partner-modulated risk-taking. No training, no neural nets, instant results.

## Why this might be the right frame

**Trust is a gain modulator, not a learned policy.** Real mice don't learn a new policy when near a sibling. They adjust the gain on existing hardwired circuits — oxytocin modulates the amygdala's fear response, it doesn't rewrite the motor cortex. The PID framing maps directly onto this neuroscience: trust = coefficient on the predator-repulsion term.

**Interpretability.** When an RL agent changes behavior, you can't cleanly attribute it to trust. With PID, trust literally *is* β. Full analytical characterization of the trajectory as a function of the parameter.

**Speed.** Exhaustive parameter sweeps and phase diagrams in minutes, not hours of training.

## The three-level hierarchy

| Level | Tool | What it provides |
|-------|------|-----------------|
| Mathematical theory | PID | Trust = gain modulation. Analytical predictions. Phase diagrams. |
| Behavioral question | LLMs | Do language priors about trust produce the correct gain changes? |
| Ground truth | Real mice | Sibling vs non-sibling trajectory data |

These aren't competing methods. Each operates at a different level of analysis.

## Using PID as an interpretive lens on LLMs

Key experiment: fit PID parameters to LLM-generated trajectories.
- Run LLM with "sibling" prompt → get trajectory → fit (risk_tolerance, partner_trust) that best reproduces it
- Run LLM with "stranger" prompt → fit parameters again
- Question: does the fitted `partner_trust` coefficient change in the expected direction?
- Extension: do same-family model pairs (Qwen+Qwen) yield higher fitted trust than cross-family (Qwen+Gemma)?

This gives a quantitative, interpretable measure of what trust prompts actually *do* to LLM behavior.

## Where RL still matters

The "forcing function" question: when does reactive control (PID) become insufficient and planning emerge? PID can't do multi-step lookahead, deception, or baiting. If we increase predator intelligence or add obstacles that require sequential decisions, at what point does a planning agent outperform the PID? And does trust shift that transition point?

That's the deeper theoretical contribution — but it's a separate paper from the trust-modulates-risk empirical finding.

## What to build next
1. Parameter fitting: given a trajectory, find the PID coefficients that minimize reconstruction error
2. Systematic sweep: risk × trust × predator_speed grid, compute all metrics
3. LLM→PID mapping: fit PID params to LLM trajectories, report fitted trust per condition
