# Cellworld Gymnasium Environments

Base repo: https://github.com/germanespinosa/cellworld_game

Two Gymnasium-compatible environments wrapping the Cellworld simulation:

| Environment | File | Task |
|---|---|---|
| `BotEvadeEnv` | `botevade_gym.py` | Prey evades a predator robot to reach a single goal |
| `OasisEnv` | `oasis_gym.py` | Prey visits a sequence of goal locations while avoiding a predator |

---

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

| Parameter | Type | Default | Description |
|---|---|---|---|
| `world_name` | `str` | `"oasis_island7_02"` | Cellworld layout name |
| `goal_locations` | `list[tuple]` | 7 canonical oases | (x, y) goal positions |
| `use_predator` | `bool` | `True` | Enable the predator robot |
| `use_lppos` | `bool` | `False` | Use TLPPO action list instead of full list |
| `max_step` | `int` | `500` | Truncation step limit |
| `reward_function` | `callable` | `lambda obs: 0` | Custom reward `f(obs) -> float` |
| `time_step` | `float` | `0.25` | Simulated seconds per `env.step()` call |
| `render` | `bool` | `False` | Open a pygame window |
| `real_time` | `bool` | `False` | Throttle simulation to real time |
| `point_of_view` | `PointOfView` | `TOP` | Camera view (`TOP`, `PREY`, `PREDATOR`) |
| `observation_type` | `ObservationType` | `DATA` | `DATA` (vector) or `PIXELS` (image) |
| `action_type` | `ActionType` | `DISCRETE` | `DISCRETE` (index) or `CONTINUOUS` (x,y) |
| `frame_stack_k` | `int` | `3` | Number of frames to stack |
| `puff_cool_down_time` | `float` | `0.5` | Seconds between predator puffs |
| `puff_threshold` | `float` | `0.1` | Distance at which predator puffs |
| `goal_threshold` | `float` | `0.025` | Distance at which prey "arrives" at goal |
| `goal_time` | `float` | `1.0` | Dwell time required at each goal (seconds) |
| `max_line_of_sight_distance` | `float` | `1.0` | Maximum vision range |

### Observation space (`DATA` mode)

The observation vector has shape `(10 * frame_stack_k + 7,)`.

**Stacked fields** (repeated for each frame in the stack):

| Field | Description |
|---|---|
| `prey_x`, `prey_y` | Prey position in [0, 1] |
| `prey_direction` | Prey heading in [0, 2π) |
| `predator_visible` | 1 if predator is in line-of-sight, else 0 |
| `predator_x`, `predator_y` | Predator position (0 if not visible) |
| `predator_direction` | Predator heading (0 if not visible) |
| `near_wall` | 1 if prey is near the arena wall |
| `near_occlusion` | 1 if prey is near an occlusion |
| `time_prey_seen_predator` | Step index when predator was last seen (−1 = never) |

**Non-stacked fields** (current frame only):

| Field | Description |
|---|---|
| `puffed` | 1 if puffed this step |
| `puff_cooled_down` | Remaining puff cooldown (seconds) |
| `finished` | 1 if episode ended naturally |
| `prey_goal_distance` | Distance to active goal |
| `goal_x`, `goal_y` | Active goal coordinates |
| `goals_remaining` | Goals left in the sequence |

### Action space

| `action_type` | Space | Description |
|---|---|---|
| `DISCRETE` | `Discrete(N)` | Index into the pre-computed action list |
| `CONTINUOUS` | `Box(0, 1, (2,))` | `(x, y)` destination in canonical coordinates |

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

## Rendering the Oasis environment

```bash
# Default: 2 episodes with predator, discrete actions
python render_oasis.py

# No predator
python render_oasis.py --no-predator

# Continuous action space
python render_oasis.py --continuous

# Run 5 episodes
python render_oasis.py --episodes 5
```

A pygame window opens showing the arena, prey (mouse), predator (robot),
goal oases (green = active, red = inactive), and the predator puff radius.

---

## Training with SAC

```bash
python SAC_train.py --config configs/sac_peeking_0406.yaml
```

Edit `configs/configs.py` and run it to generate a new config with today's date:

```bash
python configs/configs.py
python SAC_train.py --config configs/sac_peeking_<MMDD>.yaml
```
