# LLM Experiments: Trust and Risk in Language Agents

## Why LLMs
- No training needed — priors about trust/siblings/risk come from pretraining
- Cross-model pairings are a free manipulation of "familiarity"
- Directly extends the ICML Section 7 finding (single LLM was reckless)
- New question: does social context modulate that recklessness?

## Models (via OpenRouter)
Frontier: Gemini Pro 3.1, Claude Opus 4.6 / Sonnet 4.6, GPT-5.4
Smaller/tunable: Qwen 35B, Gemma 4, Kimi v2

## Architecture
```
Game State → Text Serializer → LLM → Action Parser → DualEvade env
                                ↑
                          System prompt (trust condition)
                          Episode history (trust emergence)
```

LLMs act as **planners** (choose waypoint every N sim-steps), not reactive controllers. This matches:
- API latency constraints
- How mice actually decide (deliberate navigation, not muscle twitches)
- The ICML paper's LLM protocol

## Experiment 1: Prompted trust (within-model)
Same model controls both mice.
- **Sibling prompt**: "Mouse 2 is your sibling. You've navigated this arena together many times before."
- **Stranger prompt**: "Mouse 2 is unknown to you. You have no information about their behavior."
- **Control**: No mention of the other mouse's relationship.

Measure: Do sibling-prompted agents enter high-exposure zones more? Skip waiting? Take shorter paths?

## Experiment 2: Cross-model familiarity (no prompting)
- **Same-family**: Qwen 35B + Qwen 35B, Gemma + Gemma, Claude + Claude
- **Cross-family**: Qwen + Gemma, Claude + GPT, etc.
- Neutral prompts only — no trust language.

Hypothesis: Same-family pairs share behavioral priors from similar training distributions → naturally higher coordination → more risk-taking. Cross-family pairs have mismatched priors → less coordination → conservative play.

This is the cleanest experiment because there's zero prompt engineering confound.

## Experiment 3: Trust emergence over episodes
Two LLMs play repeated episodes. After each episode, both receive a summary:
"Episode 4 result: Mouse 2 waited at the entrance for 3 steps, then followed the wall to the goal. Mouse 2 was not captured."

Does risk-taking increase over episodes as context about partner accumulates?
Does it increase faster for same-model pairs?

## Experiment 4: Comparison to real mouse data
Use the same metrics from the ICML paper:
- Waiting behavior %
- Thigmotaxis %
- Visitation density overlap
- Episode length distribution
- Distance-to-predator CDF

Compare LLM sibling condition → real sibling mice, LLM stranger condition → real stranger mice.

## Text state format
```
You are Mouse 1 in a hexagonal arena. Your goal: reach (1.0, 0.5).
A predator is hunting you. If it gets within 0.1 units, you are captured.

Current state:
- Your position: (0.23, 0.48), facing 15°
- Mouse 2 position: (0.31, 0.52), facing 340°
- Predator: visible at (0.65, 0.40), facing 200°
- Distance to goal: 0.78
- Distance to predator: 0.43
- Distance to Mouse 2: 0.09
- Near wall: yes
- Near obstacle: no

Respond with a JSON action: {"x": float, "y": float, "wait": bool}
```

## API details
- All calls through OpenRouter (key in .env)
- Rate limiting and retry logic built in
- Structured output parsing with fallbacks
- Cost tracking per experiment
