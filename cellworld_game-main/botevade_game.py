"""Minimal BotEvade demo driving the prey with the new point-mass dynamics
via `MousePIDController`. The prey's action is `(ax, ay)`; the controller
converts "go to this waypoint" into a `(ax, ay)` command every sim step."""
import random
from cellworld_game import (
    BotEvade,
    MousePIDController,
    save_video_output,
    save_log_output,
)

bot_evade = BotEvade(world_name="00_00",
                     puff_cool_down_time=.5,
                     puff_threshold=.1,
                     goal_threshold=.05,
                     time_step=.025,
                     real_time=False,
                     render=True,
                     use_predator=True,
                     predator_prey_forward_speed_ratio=1.5,
                     predator_prey_turning_speed_ratio=1.5)


def puff_processing(model):
    print("you've been puffed")


bot_evade.add_event_handler(event_name="puff",
                            handler=puff_processing)


save_video_output(bot_evade, "videos")
save_log_output(bot_evade, "test", "logs")
bot_evade.reset()
bot_evade.view.show_sprites = False

prey_pid = MousePIDController(bot_evade.prey)

last_destination_time = -3
random_actions = 10

while bot_evade.running:
    if bot_evade.time > last_destination_time + 2:
        if bot_evade.prey_data.goal_achieved or random_actions == 0:
            destination = bot_evade.goal_location
            random_actions = 10
        else:
            random_actions -= 1
            destination = random.choice(bot_evade.loader.open_locations)
        prey_pid.set_destination(destination)
        last_destination_time += 2
    prey_pid.step()
    bot_evade.step()
