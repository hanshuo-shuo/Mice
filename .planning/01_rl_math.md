# RL Formulation: Trust as Partner Predictability

## Setup
Two prey agents in the DualEvade cellworld environment with a reactive predator.

Each agent i has:
- Policy π_i(a|s)
- Partner model π̂_j (agent i's prediction of agent j's policy)
- Trust metric: ε = E[||π̂_j - π_j||] (partner prediction error, inverse trust)

## Why reward sharing isn't enough
A shared-reward slider is trivially engineered — you built what you measured. Reviewers will reject this.

Instead: trust should be **emergent from co-training**, not hand-coded. The claim is that agents who can better predict each other's behavior naturally take more risks. The reward structure stays fixed; what varies is partner familiarity.

## Sibling condition (co-trained)
- Both agents trained together in the same environment
- Each develops an implicit model of the other's policy through shared experience
- High mutual predictability → low ε → more risk-taking

## Stranger condition (independently trained)
- Agents trained separately, paired at test time
- No model of partner's policy → high ε → conservative play
- Partner is effectively additional stochasticity

## Variance-penalized multi-agent TD target
Building on VP-TDMPC-2 from the ICML paper:

```
TD_target_i = r_t + γ(Q_i(s_{t+1}, π_i(s_{t+1})) - α·Var_a(Q_i(s_{t+1}, a)) - β·ε_i)
```

Where:
- α·Var penalizes uncertain states (fear of death, from ICML paper)
- β·ε penalizes unpredictable partners (trust modulation, new)

When ε is low (sibling), the β·ε term vanishes → agent behaves as if alone with a predictable ally.
When ε is high (stranger), the penalty increases → agent is more conservative.

## Partner prediction module
Agent i maintains a small network f_i that predicts agent j's next action given current state:

```
â_j = f_i(s_t)
ε_i = ||â_j - a_j||² (rolling average over window W)
```

This is learned online during co-training. For the stranger condition, f_i is not trained (or trained on a different partner), so ε remains high.

## Phase diagram (aspirational)
Vary systematically:
- V: observation range
- C: predator speed/intelligence
- N: number of agents
- co-training duration (proxy for trust)

Map out where in (V, C, N, trust) space:
1. Planning behavior (waiting, peeking) emerges
2. Trust-modulated risk-taking emerges
3. Cooperative strategies (baiting, leading) emerge

This is the "psychohistory" result — mathematical conditions for behavioral phase transitions.

## Practical plan
1. Extend DualEvade env to multi-agent Gymnasium wrapper
2. Port VP-TDMPC-2 to multi-agent (each agent gets own policy + partner model)
3. Train sibling pairs (co-trained) and stranger pairs (independently trained, paired at eval)
4. Measure risk metrics across conditions
