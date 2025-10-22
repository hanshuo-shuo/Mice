import enum
import typing
import cellworld_game as cwgame
import numpy as np
import math
from gymnasium import Env
from gymnasium import spaces
from gymnasium.envs.registration import register
from enum import Enum


class Observation(np.ndarray):
    fields = []  # list of field names in the observation

    def __init__(self):
        super().__init__()
        for index, field in enumerate(self.__class__.fields):
            self._create_property(index=index,
                                  field=field)
        self.field_enum = Enum("fields", {field: index for index, field in enumerate(self.__class__.fields)})

    def __new__(cls):
        # Create a new array of zeros with the given shape and dtype
        shape = (len(cls.fields),)
        dtype = np.float32
        buffer = None
        offset = 0
        strides = None
        order = None
        obj = super(Observation, cls).__new__(cls, shape, dtype, buffer, offset, strides, order)
        obj.fill(0)
        return obj

    def _create_property(self,
                         index: int,
                         field: str):
        def getter(self):
            return self[index]

        def setter(self, value):
            self[index] = value

        setattr(self.__class__, field, property(getter, setter))

    def __setitem__(self, field: typing.Union[Enum, int], value):
        if isinstance(field, Enum):
            np.ndarray.__setitem__(self, field.value, value)
        else:
            np.ndarray.__setitem__(self, field, value)

    def __getitem__(self, field: typing.Union[Enum, int]) -> np.ndarray:
        if isinstance(field, Enum):
            return np.ndarray.__getitem__(self, field.value)
        else:
            return np.ndarray.__getitem__(self, field)


class Environment(Env):
    def __init__(self):
        self.event_handlers: typing.Dict[str, typing.List[typing.Callable]] = {"reset": [],
                                                                               "step": []}

    def __handle_event__(self, event_name: str, *args):
        for handler in self.event_handlers[event_name]:
            handler(*args)

    def add_event_handler(self, event_name: str, handler: typing.Callable):
        if event_name not in self.event_handlers:
            raise "Event handler not registered"
        self.event_handlers[event_name].append(handler)

    def reset(self,
              options: typing.Optional[dict] = None,
              seed=None):
        self.__handle_event__("reset", options, seed)

    def step(self, action: int):
        self.__handle_event__("step", action)


class OasisObservation(Observation):
    """
    Observation for Oasis environment.
    
    Fields:
    - prey_x, prey_y: Prey position
    - prey_direction: Prey heading direction (radians)
    - predator_x, predator_y: Predator position (0,0 if not visible)
    - predator_direction: Predator heading (radians, 0 if not visible)
    - goal_x, goal_y: Current goal location
    - prey_goal_distance: Distance from prey to current goal
    - goal_achieved: Whether prey is at goal accumulating time (not completed yet!)
    - goal_just_completed: Whether a goal was just completed this step (1 for one step only)
    - puffed: Whether prey was just captured this step
    - puff_cooled_down: Whether puff cooldown has expired
    - goals_remaining: Number of goals left to visit in sequence
    - finished: Whether episode is complete
    """
    fields = ["prey_x",
              "prey_y",
              "prey_direction",
              "predator_x",
              "predator_y",
              "predator_direction",
              "goal_x",
              "goal_y",
              "prey_goal_distance",
              "goal_achieved",
              "goal_just_completed",
              "puffed",
              "puff_cooled_down",
              "goals_remaining",
              "finished"]


class OasisEnv(Environment):
    """
    Gymnasium wrapper for the Oasis game environment.
    
    In this environment:
    - The prey (mouse) must visit a sequence of goal locations
    - The prey must stay at each goal for a specified duration to achieve it
    - After visiting all goals, the prey must return to the start location
    - An optional predator (robot) chases the prey
    - If the predator catches the prey (within puff_threshold), a "puff" occurs
    - Episode ends when all goals are achieved and prey returns to start
    """

    PointOfView = cwgame.Oasis.PointOfView
    AgentRenderMode = cwgame.Agent.RenderMode

    class ObservationType(enum.Enum):
        DATA = 0
        PIXELS = 1

    class ActionType(enum.Enum):
        DISCRETE = 0
        CONTINUOUS = 1

    def __init__(self,
                 world_name: str,
                 goal_locations: typing.List[typing.Tuple[float, float]],
                 goal_sequence_generator: typing.Callable[[None], typing.List[int]] = None,
                 use_lppos: bool = True,
                 use_predator: bool = True,
                 max_step: int = 500,
                 reward_function: typing.Callable[[OasisObservation], float] = lambda x: 0,
                 time_step: float = .25,
                 model_time_step: float = .025,
                 puff_cool_down_time: float = .5,
                 goal_time: float = .25,
                 puff_threshold: float = .1,
                 goal_threshold: float = .1,
                 render: bool = False,
                 real_time: bool = False,
                 point_of_view: PointOfView = PointOfView.TOP,
                 agent_render_mode: AgentRenderMode = AgentRenderMode.SPRITE,
                 observation_type: ObservationType = ObservationType.DATA,
                 action_type: ActionType = ActionType.DISCRETE,
                 predator_speed_multiplier: float = 1.0,
                 goal_conditioned: bool = False):
        """
        Initialize the Oasis environment.
        
        Args:
            world_name: Name of the world configuration
            goal_locations: List of goal locations (x, y) tuples
            goal_sequence_generator: Function that returns list of goal indices to visit
            use_lppos: Whether to use limited action list (LPPO actions)
            use_predator: Whether to include predator
            max_step: Maximum steps per episode
            reward_function: Function to calculate reward from observation
            time_step: Time between RL steps (action frequency) in seconds
            model_time_step: Time step for physics simulation in seconds
            puff_cool_down_time: Cooldown time after a puff
            goal_time: Time required at goal to achieve it
            puff_threshold: Distance threshold for puff
            goal_threshold: Distance threshold for goal achievement
            render: Whether to render visualization
            real_time: Whether to run in real time
            point_of_view: Camera perspective
            agent_render_mode: How agents are rendered
            observation_type: DATA or PIXELS observation
            action_type: DISCRETE or CONTINUOUS actions
            predator_speed_multiplier: Speed multiplier for predator (default 1.0, <1.0 slower, >1.0 faster)
            goal_conditioned: If True, use goal-conditioned RL format (dict observation with achieved/desired goal)
        """
        if observation_type == OasisEnv.ObservationType.PIXELS and not render:
            raise ValueError("Cannot use PIXELS observation type without render")
        
        if observation_type == OasisEnv.ObservationType.PIXELS and goal_conditioned:
            raise ValueError("Goal-conditioned mode only supports DATA observation type")

        self.max_step = max_step
        self.reward_function = reward_function
        self.time_step = time_step
        self.model_time_step = model_time_step
        self.goal_locations = goal_locations
        self.goal_conditioned = goal_conditioned
        self.loader = cwgame.CellWorldLoader(world_name=world_name)

        # Set up action space
        if use_lppos:
            self.action_list = self.loader.tlppo_action_list
        else:
            self.action_list = self.loader.full_action_list

        self.action_type = action_type
        if self.action_type == OasisEnv.ActionType.DISCRETE:
            self.action_space = spaces.Discrete(len(self.action_list))
        else:
            self.action_space = spaces.Box(0.0, 1.0, (2,), dtype=np.float32)

        # Create the Oasis model
        self.model = cwgame.Oasis(
            world_name=world_name,
            goal_locations=goal_locations,
            goal_sequence_generator=goal_sequence_generator,
            use_predator=use_predator,
            puff_cool_down_time=puff_cool_down_time,
            goal_time=goal_time,
            puff_threshold=puff_threshold,
            goal_threshold=goal_threshold,
            time_step=model_time_step,  # Use smaller physics time step
            real_time=real_time,
            render=render,
            point_of_view=point_of_view,
            agent_render_mode=agent_render_mode
        )
        
        # Adjust predator speed if needed
        if use_predator and predator_speed_multiplier != 1.0:
            self.model.predator.max_forward_speed *= predator_speed_multiplier
            self.model.predator.max_turning_speed *= predator_speed_multiplier

        # Set up observation space
        self.observation_type = observation_type
        if self.observation_type == OasisEnv.ObservationType.DATA:
            self.observation = OasisObservation()
            
            if self.goal_conditioned:
                # Goal-conditioned observation space (Dict)
                # observation: prey state without goal info
                # achieved_goal: current prey position (x, y)
                # desired_goal: target goal position (x, y)
                obs_dim = len(self.observation) - 5  # Remove goal_x, goal_y, prey_goal_distance, goals_remaining, goal_just_completed
                self.observation_space = spaces.Dict({
                    "observation": spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32),
                    "achieved_goal": spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32),
                    "desired_goal": spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
                })
            else:
                # Standard observation space
                self.observation_space = spaces.Box(-np.inf, np.inf, self.observation.shape, dtype=np.float32)
        else:
            self.observation = self.model.view.get_screen(normalized=True)
            self.observation_space = spaces.Box(0.0, 1.0, self.observation.shape, dtype=np.float32)

        self.episode_reward = 0
        self.step_count = 0
        self.previous_goals_remaining = 0
        Environment.__init__(self)

    def __update_observation__(self):
        """Update the observation based on current model state."""
        if self.observation_type == OasisEnv.ObservationType.DATA:
            # Prey state
            self.observation.prey_x = self.model.prey.state.location[0]
            self.observation.prey_y = self.model.prey.state.location[1]
            self.observation.prey_direction = math.radians(self.model.prey.state.direction)

            # Predator state (if used and visible)
            if self.model.use_predator:
                # Check if predator is visible (has line of sight)
                predator_visible = self.model.visibility.line_of_sight(
                    self.model.prey.state.location,
                    self.model.predator.state.location
                )
                if predator_visible:
                    self.observation.predator_x = self.model.predator.state.location[0]
                    self.observation.predator_y = self.model.predator.state.location[1]
                    self.observation.predator_direction = math.radians(self.model.predator.state.direction)
                else:
                    self.observation.predator_x = 0
                    self.observation.predator_y = 0
                    self.observation.predator_direction = 0
            else:
                self.observation.predator_x = 0
                self.observation.predator_y = 0
                self.observation.predator_direction = 0

            # Goal state
            if self.model.goal_location is not None:
                self.observation.goal_x = self.model.goal_location[0]
                self.observation.goal_y = self.model.goal_location[1]
            else:
                # When goal_location is None, task is complete
                # Set goal to current prey position to avoid misleading signal
                self.observation.goal_x = self.model.prey.state.location[0]
                self.observation.goal_y = self.model.prey.state.location[1]

            self.observation.prey_goal_distance = self.model.prey_goal_distance
            self.observation.goal_achieved = 1 if self.model.goal_achieved else 0
            self.observation.goal_just_completed = 0  # Will be set in __step__
            self.observation.puffed = 1 if self.model.puffed else 0
            self.observation.puff_cooled_down = 1 if self.model.puff_cool_down <= 0 else 0
            self.observation.goals_remaining = len(self.model.goal_sequence)
            self.observation.finished = not self.model.running
            
            if self.goal_conditioned:
                # Return goal-conditioned format
                # observation: prey state + predator state + goal_achieved + puffed + puff_cooled_down + finished
                # Remove: goal_x, goal_y, prey_goal_distance, goals_remaining, goal_just_completed
                obs_array = np.array([
                    self.observation.prey_x,
                    self.observation.prey_y,
                    self.observation.prey_direction,
                    self.observation.predator_x,
                    self.observation.predator_y,
                    self.observation.predator_direction,
                    self.observation.goal_achieved,
                    self.observation.puffed,
                    self.observation.puff_cooled_down,
                    self.observation.finished
                ], dtype=np.float32)
                
                achieved_goal = np.array([
                    self.model.prey.state.location[0],
                    self.model.prey.state.location[1]
                ], dtype=np.float32)
                
                desired_goal = np.array([
                    self.observation.goal_x,
                    self.observation.goal_y
                ], dtype=np.float32)
                
                return {
                    "observation": obs_array,
                    "achieved_goal": achieved_goal,
                    "desired_goal": desired_goal
                }
            else:
                return self.observation
        else:
            # Pixel-based observation
            self.observation = self.model.view.get_screen(normalized=True)
            return self.observation

    def set_action(self, action: typing.Union[int, typing.Tuple[float, float]]):
        """Set the action for the prey agent."""
        if self.action_type == OasisEnv.ActionType.DISCRETE:
            self.model.prey.set_destination(self.action_list[action])
        else:
            self.model.prey.set_destination(tuple(action))
    
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> float:
        """
        Compute the reward for goal-conditioned RL.
        This is ONLY called by HER replay buffer for hindsight relabeling, NOT in env.step()!
        
        Args:
            achieved_goal: Current position (x, y) - can be batched
            desired_goal: Target position (x, y) - can be batched
            info: Additional info dict
            
        Returns:
            Sparse reward for HER: 0 if at goal, -1 otherwise
        """
        # Compute distance to goal (supports batched inputs)
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        
        # Standard sparse reward for HER: 0 if success, -1 otherwise
        # This is the recommended reward format for HER
        reward = -(distance > self.model.goal_threshold).astype(np.float32)
        
        return reward

    def __step__(self):
        """Internal step function that updates observation and computes reward."""
        self.step_count += 1
        truncated = (self.step_count >= self.max_step)
        obs = self.__update_observation__()
        
        # Detect if a goal was just completed
        current_goals_remaining = len(self.model.goal_sequence)
        goal_just_completed = (self.previous_goals_remaining > current_goals_remaining)
        self.previous_goals_remaining = current_goals_remaining
        
        # Compute reward using the user-defined reward function
        # Note: In goal-conditioned mode, we need to convert dict obs to OasisObservation for reward_function
        if self.goal_conditioned:
            # Create a temporary OasisObservation for reward calculation
            temp_obs = self.observation  # This is the full OasisObservation
            temp_obs.goal_just_completed = 1 if goal_just_completed else 0
            reward = self.reward_function(temp_obs)
        else:
            # Standard mode: update goal_just_completed in obs
            obs.goal_just_completed = 1 if goal_just_completed else 0
            reward = self.reward_function(obs)
        
        self.episode_reward += reward

        # Reset puffed flag after step (it's only true for one step)
        if self.model.puffed:
            self.model.puffed = False

        # Check if episode is done
        terminated = not self.model.running

        # Build info dictionary
        if terminated or truncated:
            # Episode is complete, compute success metrics
            # Success: completed all goals and returned to start with no or few puffs
            is_success = terminated and self.model.goal_location is None
            info = {
                "puff_count": self.model.puff_count,
                "reward": self.episode_reward,
                "is_success": 1 if is_success else 0,
                "goals_completed": len(self.goal_locations) - len(self.model.goal_sequence),
                "goal_just_completed": goal_just_completed,
                "agents": {}
            }
        else:
            info = {"goal_just_completed": goal_just_completed}

        return obs, reward, terminated, truncated, info

    def replay_step(self, agents_state: typing.Dict[str, cwgame.AgentState]):
        """
        Step the environment using provided agent states (for replay).
        
        Args:
            agents_state: Dictionary mapping agent names to their states
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.model.set_agents_state(agents_state=agents_state,
                                    delta_t=self.model_time_step)
        return self.__step__()

    def step(self, action: typing.Union[int, typing.Tuple[float, float]]):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (discrete index or continuous tuple)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.set_action(action=action)
        model_t = self.model.time + self.time_step
        while self.model.running and self.model.time < model_t:
            self.model.step()
        Environment.step(self, action=action)
        return self.__step__()

    def __reset__(self):
        """Internal reset function."""
        self.episode_reward = 0
        self.step_count = 0
        self.previous_goals_remaining = len(self.model.goal_sequence)
        return self.__update_observation__(), {}

    def reset(self,
              options: typing.Optional[dict] = None,
              seed=None):
        """
        Reset the environment to initial state.
        
        Args:
            options: Optional reset options
            seed: Random seed
            
        Returns:
            Tuple of (observation, info)
        """
        self.model.reset()
        Environment.reset(self, options=options, seed=seed)
        return self.__reset__()

    def replay_reset(self, agents_state: typing.Dict[str, cwgame.AgentState]):
        """
        Reset the environment using provided agent states (for replay).
        
        Args:
            agents_state: Dictionary mapping agent names to their states
            
        Returns:
            Tuple of (observation, info)
        """
        self.model.reset()
        self.model.set_agents_state(agents_state=agents_state)
        return self.__reset__()

    def close(self):
        """Close the environment and clean up resources."""
        self.model.close()
        Env.close(self=self)

