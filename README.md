## Cellworld Game — Single-Prey RL (BotEvade / Oasis)

Base repo: [germanespinosa/cellworld_game](https://github.com/germanespinosa/cellworld_game)

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
