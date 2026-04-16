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

## OasisEnv

### Task description

The prey starts at the left entrance `(0.05, 0.5)` and must visit a randomly
sampled sequence of **goal oases**, dwelling at each one for `goal_time` seconds,
then return to the start. A predator robot chases the prey and can "puff" it
(penalty event) when close and in line-of-sight.

### Quick start

```python
from oasis_gym import OasisEnv

env = OasisEnv(
    world_name="oasis_island7_02",
    use_predator=True,
    render=False,
)

obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    done = done or truncated

env.close()
```

### Constructor parameters


| Parameter                    | Type              | Default              | Description                                |
| ---------------------------- | ----------------- | -------------------- | ------------------------------------------ |
| `world_name`                 | `str`             | `"oasis_island7_02"` | Cellworld layout name                      |
| `goal_locations`             | `list[tuple]`     | 7 canonical oases    | (x, y) goal positions                      |
| `use_predator`               | `bool`            | `True`               | Enable the predator robot                  |
| `use_lppos`                  | `bool`            | `False`              | Use TLPPO action list instead of full list |
| `max_step`                   | `int`             | `500`                | Truncation step limit                      |
| `reward_function`            | `callable`        | `lambda obs: 0`      | Custom reward `f(obs) -> float`            |
| `time_step`                  | `float`           | `0.25`               | Simulated seconds per `env.step()` call    |
| `render`                     | `bool`            | `False`              | Open a pygame window                       |
| `real_time`                  | `bool`            | `False`              | Throttle simulation to real time           |
| `point_of_view`              | `PointOfView`     | `TOP`                | Camera view (`TOP`, `PREY`, `PREDATOR`)    |
| `observation_type`           | `ObservationType` | `DATA`               | `DATA` (vector) or `PIXELS` (image)        |
| `action_type`                | `ActionType`      | `DISCRETE`           | `DISCRETE` (index) or `CONTINUOUS` (x,y)   |
| `frame_stack_k`              | `int`             | `3`                  | Number of frames to stack                  |
| `puff_cool_down_time`        | `float`           | `0.5`                | Seconds between predator puffs             |
| `puff_threshold`             | `float`           | `0.1`                | Distance at which predator puffs           |
| `goal_threshold`             | `float`           | `0.025`              | Distance at which prey "arrives" at goal   |
| `goal_time`                  | `float`           | `1.0`                | Dwell time required at each goal (seconds) |
| `max_line_of_sight_distance` | `float`           | `1.0`                | Maximum vision range                       |


### Observation space (`DATA` mode)

The observation vector has shape `(10 * frame_stack_k + 7,)`.

**Stacked fields** (repeated for each frame in the stack):


| Field                      | Description                                         |
| -------------------------- | --------------------------------------------------- |
| `prey_x`, `prey_y`         | Prey position in [0, 1]                             |
| `prey_direction`           | Prey heading in [0, 2π)                             |
| `predator_visible`         | 1 if predator is in line-of-sight, else 0           |
| `predator_x`, `predator_y` | Predator position (0 if not visible)                |
| `predator_direction`       | Predator heading (0 if not visible)                 |
| `near_wall`                | 1 if prey is near the arena wall                    |
| `near_occlusion`           | 1 if prey is near an occlusion                      |
| `time_prey_seen_predator`  | Step index when predator was last seen (−1 = never) |


**Non-stacked fields** (current frame only):


| Field                | Description                       |
| -------------------- | --------------------------------- |
| `puffed`             | 1 if puffed this step             |
| `puff_cooled_down`   | Remaining puff cooldown (seconds) |
| `finished`           | 1 if episode ended naturally      |
| `prey_goal_distance` | Distance to active goal           |
| `goal_x`, `goal_y`   | Active goal coordinates           |
| `goals_remaining`    | Goals left in the sequence        |


### Custom reward example

```python
def oasis_reward(obs):
    # obs is a flat numpy array; non-stacked fields are at the end
    puffed        = obs[-7]   # index of puffed in non-stack block
    goal_distance = obs[-4]   # prey_goal_distance
    finished      = obs[-5]

    reward = -0.01            # small time penalty
    if puffed > 0:
        reward -= 1.0
    if finished > 0:
        reward += 5.0         # bonus for completing all goals
    return reward

env = OasisEnv(reward_function=oasis_reward)
```

---

## BotEvadeEnv

### Quick start

```python
from botevade_gym import BotEvadeEnv
from reward import custom_reward

env = BotEvadeEnv(
    world_name="clump01_05",
    use_lppos=False,
    use_predator=True,
    reward_function=custom_reward,
)

obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    done = done or truncated

env.close()
```

---

## Training with SAC

```bash
python SAC_train_example.py --config configs/sac_oasis_0416.yaml
```

Edit `configs/configs.py` and run it to generate a new config with today's date:

```bash
python configs/configs.py
python SAC_train_example.py --config configs/NAMEOFCONFIG_<MMDD>.yaml
```

## TO DO

- **Oasis map navigation bug**: `Navigation.get_path()` generates paths that cut through obstacles on non-21_05 maps (e.g. `oasis_island7_02`). The visibility-based path optimization skips waypoints assuming line-of-sight, but the resulting straight-line segments can clip obstacle geometry. Need to either fix the visibility graph or add collision as a fallback.
- **Oasis observation incomplete**: The current observation space is missing data. Previously we had 21_05-specific data for `near_obstacle` and `near_wall` observations — this needs to be generalized for other maps like `oasis_island7_02`.

