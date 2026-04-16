## Cellworld Game — Single-Prey RL (BotEvade / Oasis)

Base repo: [germanespinosa/cellworld_game](https://github.com/germanespinosa/cellworld_game)

---

# Cellworld Gymnasium Environments

Base repo: [https://github.com/germanespinosa/cellworld_game](https://github.com/germanespinosa/cellworld_game)

Two Gymnasium-compatible environments wrapping the Cellworld simulation:


| Environment   | File              | Task                                                               |
| ------------- | ----------------- | ------------------------------------------------------------------ |
| `BotEvadeEnv` | `botevade_gym.py` | Prey evades a predator robot to reach a single goal                |
| `OasisEnv`    | `oasis_gym.py`    | Prey visits a sequence of goal locations while avoiding a predator |


---

## Environment Description

Both environments are **POMDPs**: the prey agent does not always have line-of-sight to the predator, so the observation is only a partial view of the true state. This has practical implications for algorithm choice:

- **Model-free methods** (e.g., SAC, PPO) work fine out of the box; frame stacking (`frame_stack_k`) provides a basic temporal context.
- **Model-based methods** should account for partial observability — a recurrent world model (e.g., LSTM-based) is recommended to maintain a belief state over the hidden predator position.

## RL Resources


| Library           | Link                                                                                                                             | Notes                                                                |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Spinning Up       | [https://spinningup.openai.com/en/latest/user/introduction.html](https://spinningup.openai.com/en/latest/user/introduction.html) | Good conceptual intro to RL algorithms                               |
| Stable-Baselines3 | [https://stable-baselines3.readthedocs.io/en/master/index.html](https://stable-baselines3.readthedocs.io/en/master/index.html)   | Easy to use; covers standard model-free algorithms                   |
| Tianshou          | [https://tianshou.org/en/stable/](https://tianshou.org/en/stable/)                                                               | More flexible; supports n-step returns and custom collectors         |
| SheepRL           | [https://github.com/Eclectic-Sheep/sheeprl](https://github.com/Eclectic-Sheep/sheeprl)                                           | Model-based RL; includes Dreamer-V3 with a built-in LSTM world model |


## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yaml
conda activate Mice-BotEvade
```

### 2. Install the cellworld_game package

The simulation lives in the bundled `cellworld_game-main/` folder.
No extra install step is needed — both gym files add it to `sys.path` automatically.

### 3. Verify the install

```bash
python -c "import cellworld_game; print('OK')"
```

---

### Design notes

#### Prey dynamics: PointMaze-style (ax, ay)

Prey use 2D point-mass dynamics (`(ax, ay)` action, `(vx, vy)` state) matching `gymnasium-robotics/PointMaze` — semi-implicit Euler + linear damping. The old unicycle + A\* + PID (`set_destination`) is replaced; predator (`Robot`) keeps unicycle navigation. 

---

---

### Training

```bash
# SAC single-prey (BotEvade)
python SAC_train.py --config configs/sac_peeking_0406.yaml
```

### Evaluation

```bash
# Random-policy baseline on Oasis
python eval_oasis.py --episodes 20 --predator-ratio 0.15

# Render a few episodes
python eval_oasis.py --episodes 5 --render --predator-ratio 0.20

# Trained SAC checkpoint on BotEvade
python eval_sac.py --checkpoint runs/...
```

---


**`configs/`**

| File | Description |
|------|-------------|
| `sac_peeking_0406.yaml` | Single-prey SAC hyperparams. |
| `configs.py` | Config dataclasses. |
