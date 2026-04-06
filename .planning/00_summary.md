# Trust-Modulated Risk-Taking in Multi-Agent Cellworld

## One-liner
Two mice, sibling or stranger. Does higher trust lead to more risk-taking? Test with RL and LLMs.

## Motivating connection: addiction and trust
Research in addiction therapy shows that drug-related risk-taking most frequently emerges when people are in a safe social situation around others they trust. It's only after 1-2 initial safe experiences with trusted others that individuals then venture out and engage in risky behavior on their own. This maps directly onto the mouse predator-avoidance setting: does the presence of a trusted partner lower the threshold for risk-taking? The addiction literature suggests trust is a *gateway* to risk — not because trusted partners encourage risk, but because their presence reduces perceived danger enough to make risk feel viable. This has direct NIH relevance: understanding the social conditions that enable risk-taking is central to addiction research, anxiety disorders, and social neuroscience.

## Background
- **Prior work (ICML, "Of Mice and Machines")**: Single-agent RL doesn't fear death. VP-TDMPC-2 + TISB fix this for a lone mouse. LLMs (GPT-4o) also act recklessly despite claiming caution.
- **New biological observation**: Sibling mouse pairs take significantly more risks near the predator than non-sibling pairs. Trust modulates risk. This mirrors the addiction finding — safe social context enables risk.
- **This project**: Reproduce this phenomenon computationally. Two substrates: RL agents and LLM agents.

## Core thesis
Trust = ability to predict your partner's behavior. High predictability (siblings) reduces effective environmental uncertainty, enabling riskier strategies. Low predictability (strangers) forces conservative play.

## Why both RL and LLMs

| Substrate | What it tests | Strength |
|-----------|--------------|----------|
| RL | Can the math alone produce trust-modulated risk? | No confounds — pure forcing function |
| LLMs | Do models with human priors about trust behave differently? | Cross-model pairings as natural manipulation |

RL tests whether the phenomenon is a mathematical consequence of partner predictability.
LLMs test whether absorbed human priors about trust translate to behavioral differences.

## Key experiments
1. **RL co-trained vs independent**: Two VP-TDMPC-2 agents trained together (siblings) vs separately (strangers)
2. **LLM prompted trust**: Same model, "sibling" vs "stranger" system prompts
3. **LLM cross-model**: Same-family (Qwen+Qwen) vs cross-family (Qwen+Gemma) — no trust prompting
4. **Trust emergence**: Repeated episodes with accumulated context. Does risk increase over time?
5. **Comparison to real mice**: Match trajectory distributions against biological data

## Risk metrics (from ICML paper + extensions)
- Waiting behavior (movement < 0.1 units in first 6 steps)
- Thigmotaxis (% of trajectory within 0.1 units of wall)
- Visitation density overlap with real mice
- Episode length (cautious agents take longer)
- Distance-to-predator distribution
- **New**: Inter-agent distance, coordination index, solo ventures away from partner

## Deeper question
Is planning a social construct? Does trust act as a forcing function for planning emergence, alongside visual range and threat complexity? The MacIver aquatic-to-terrestrial hypothesis says planning emerged from long-range vision. Social trust may be the second forcing function: predicting your partner's policy reduces planning complexity, enabling strategies impossible alone.
