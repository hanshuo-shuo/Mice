## Cellworld Game: Trust-Modulated Risk-Taking

Base repo: https://github.com/germanespinosa/cellworld_game

### What this is

Two simulated mice navigate a hexagonal arena to reach a goal while avoiding a robotic predator. We study whether **trust** between the mice changes how much risk they take.

Real-world finding: sibling mouse pairs take far more risks near the predator than non-sibling pairs. This repo tests whether RL agents and LLM agents reproduce that phenomenon.

### Setup

```bash
# From the existing conda env
conda env create -f environment.yaml
conda activate Mice-SAC

# Additional deps for trust experiments
pip install requests python-dotenv pyyaml matplotlib
```

Put your OpenRouter key in `.env`:
```
OPENROUTER_API_KEY=sk-or-v1-...
```

### Quick start

There are three scripts, ordered from instant to full experiment:

**1. PID baseline** — No LLM, no API, runs instantly. Proves the math works.
```bash
python run_pid_baseline.py
```
Produces a 2x2 grid showing how `risk_tolerance × partner_trust` changes trajectory shape. Output: `results/pid_baseline_grid.png`

**2. Minimal LLM experiment** — One episode trust, one episode enemy, one plot.
```bash
python run_minimal.py
```
Reads `configs/trust/minimal.yaml`. Calls Gemini 3 Flash via OpenRouter. Output: `results/minimal_trust_vs_enemy.png`

**3. Full experiments** — Multiple episodes, multiple conditions, from YAML configs.
```bash
python run_trust_experiment.py --config configs/trust/sibling_vs_stranger.yaml
python run_trust_experiment.py --config configs/trust/cross_model.yaml
python run_trust_experiment.py --config configs/trust/trust_emergence.yaml
```

### Project structure

```
trust_game/                    # New multi-agent trust experiment module
  dual_evade_env.py            # Multi-agent Gymnasium wrapper (2 prey + predator)
  state_serializer.py          # Game state → text for LLMs, trust prompt variants
  metrics.py                   # Behavioral metrics (waiting, thigmotaxis, etc.)
  llm_agent.py                 # OpenRouter client, exponential backoff, never crashes
  pid_agent.py                 # PID controller with tunable risk/trust parameters
  plot.py                      # Arena + trajectory plotting

configs/trust/                 # YAML configs for each experiment
  minimal.yaml                 # Trust vs enemy, 1 episode each
  sibling_vs_stranger.yaml     # Prompted trust (sibling/stranger), 10 episodes
  cross_model.yaml             # Same-family vs cross-family model pairings
  trust_emergence.yaml         # Does trust build over repeated episodes?

.planning/                     # Research planning docs
  00_summary.md                # Project overview and thesis
  01_rl_math.md                # RL formulation of trust as partner predictability
  02_llm_experiments.md        # LLM experiment design and details

run_pid_baseline.py            # PID baseline script
run_minimal.py                 # Minimal LLM experiment script
run_trust_experiment.py        # Full experiment runner
```

### Experiments

| Experiment | Config | What it tests |
|-----------|--------|--------------|
| PID baseline | (none, runs directly) | Mathematical proof that trust modulates trajectories |
| Prompted trust | `sibling_vs_stranger.yaml` | Same LLM, sibling vs stranger system prompts |
| Cross-model | `cross_model.yaml` | Gemini+Gemini vs Qwen+Qwen vs Gemini+Qwen (no trust prompting) |
| Trust emergence | `trust_emergence.yaml` | Does risk-taking increase over repeated episodes? |

### Models used

| Model | OpenRouter ID | Role |
|-------|--------------|------|
| Gemini 3 Flash | `google/gemini-3-flash-preview` | Fast, cheap, main workhorse |
| Qwen 3.5 35B | `qwen/qwen3.5-35b-a3b` | Small open-weights, cross-family pairing |
| Claude Sonnet 4.6 | `anthropic/claude-sonnet-4.6` | Frontier control |

### Metrics

From the ICML paper ("Of Mice and Machines") plus multi-agent extensions:

- **Waiting %**: episodes where agent barely moves in first 6 steps
- **Thigmotaxis %**: fraction of trajectory near the arena wall
- **Episode length**: cautious agents take longer
- **Survival rate**: % of episodes without capture
- **Partner distance**: average distance between the two mice
- **Solo ventures %**: how often one mouse ventures far from the other

### Existing RL training (from base repo)

```bash
python SAC_train.py --config configs/sac_peeking_0406.yaml
```

### Links

- Base cellworld_game: https://github.com/germanespinosa/cellworld_game
- ICML paper: "Of Mice and Machines" (Han, Espinosa, Huang, Dombeck, MacIver, Stadie)
