import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
import torch.nn as nn

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
from stable_baselines3 import DQN

SelfDQNWithPrediction = TypeVar("SelfDQNWithPrediction", bound="DQNWithPrediction")


class StatePredictor(nn.Module):
    """
    State prediction network that predicts next state given current state and action
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256):
        super(StatePredictor, self).__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Network to predict next state
        self.predictor = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, observation_dim)
        )
        
    def forward(self, state: th.Tensor, action: th.Tensor) -> th.Tensor:
        """
        Predict next state given current state and action
        
        :param state: Current state tensor
        :param action: Action tensor (one-hot encoded for discrete actions)
        :return: Predicted next state
        """
        # Concatenate state and action
        state_action = th.cat([state, action], dim=-1)
        predicted_next_state = self.predictor(state_action)
        return predicted_next_state


class DQNWithPrediction(DQN):
    """
    DQN with state prediction capability.
    
    This extends the standard DQN to include a state prediction network that learns
    to predict the next state given current state and action. During evaluation,
    it can provide prediction errors for analysis.
    """
    
    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        # New parameters for prediction
        prediction_learning_rate: float = 1e-4,
        prediction_loss_weight: float = 1.0,
        predictor_hidden_dim: int = 256,
    ) -> None:
        
        # Initialize parent DQN
        super().__init__(
            policy, env, learning_rate, buffer_size, learning_starts, batch_size,
            tau, gamma, train_freq, gradient_steps, replay_buffer_class,
            replay_buffer_kwargs, optimize_memory_usage, target_update_interval,
            exploration_fraction, exploration_initial_eps, exploration_final_eps,
            max_grad_norm, stats_window_size, tensorboard_log, policy_kwargs,
            verbose, seed, device, False  # Don't setup model yet
        )
        
        # Prediction-specific parameters
        self.prediction_learning_rate = prediction_learning_rate
        self.prediction_loss_weight = prediction_loss_weight
        self.predictor_hidden_dim = predictor_hidden_dim
        
        # Will be initialized in _setup_model
        self.state_predictor: Optional[StatePredictor] = None
        self.predictor_optimizer: Optional[th.optim.Optimizer] = None
        
        # For tracking prediction errors during evaluation
        self.prediction_errors: List[float] = []
        self.eval_mode = False
        
        if _init_setup_model:
            self._setup_model()
    
    def _setup_model(self) -> None:
        super()._setup_model()
        
        # Get observation and action dimensions
        obs_dim = self.observation_space.shape[0]
        
        if isinstance(self.action_space, spaces.Discrete):
            action_dim = self.action_space.n
        else:
            action_dim = self.action_space.shape[0]
        
        # Initialize state predictor
        self.state_predictor = StatePredictor(
            observation_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=self.predictor_hidden_dim
        ).to(self.device)
        
        # Initialize predictor optimizer
        self.predictor_optimizer = th.optim.Adam(
            self.state_predictor.parameters(),
            lr=self.prediction_learning_rate
        )
    
    def _encode_action(self, actions: th.Tensor) -> th.Tensor:
        """
        Encode actions for the predictor network.
        For discrete actions, use one-hot encoding.
        For continuous actions, use the action values directly.
        """
        if isinstance(self.action_space, spaces.Discrete):
            # One-hot encode discrete actions
            action_dim = self.action_space.n
            batch_size = actions.shape[0]
            
            # Validate action values
            actions_long = actions.long()
            
            # Check for invalid action values
            valid_mask = (actions_long >= 0) & (actions_long < action_dim)
            if not valid_mask.all():
                invalid_actions = actions_long[~valid_mask]
                print(f"Warning: Invalid action values detected: {invalid_actions.cpu().numpy()}")
                print(f"Action space size: {action_dim}")
                print(f"Valid range: [0, {action_dim-1}]")
                # Clamp invalid actions to valid range
                actions_long = th.clamp(actions_long, 0, action_dim - 1)
            
            encoded_actions = th.zeros(batch_size, action_dim, device=self.device)
            
            # Handle different action tensor dimensions
            if actions_long.dim() == 1:
                # If actions is 1D, we need to unsqueeze for scatter
                encoded_actions.scatter_(1, actions_long.unsqueeze(1), 1)
            else:
                # If actions is already 2D, use it directly
                encoded_actions.scatter_(1, actions_long, 1)
            
            return encoded_actions
        else:
            # Use continuous actions directly
            return actions.float()
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode
        self.policy.set_training_mode(True)
        if self.state_predictor is not None:
            self.state_predictor.train()
        
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)
        
        losses = []
        prediction_losses = []
        
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            # Train Q-network (original DQN training)
            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
            
            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)
            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())
            
            # Compute Huber loss (less sensitive to outliers)
            q_loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(q_loss.item())
            
            # Optimize the Q-network
            self.policy.optimizer.zero_grad()
            q_loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            
            # Train state predictor
            if self.state_predictor is not None and self.predictor_optimizer is not None:
                # Encode actions for predictor
                encoded_actions = self._encode_action(replay_data.actions)
                
                # Predict next states
                predicted_next_states = self.state_predictor(
                    replay_data.observations, 
                    encoded_actions
                )
                
                # Compute prediction loss
                prediction_loss = F.mse_loss(predicted_next_states, replay_data.next_observations)
                prediction_losses.append(prediction_loss.item())
                
                # Optimize the predictor
                self.predictor_optimizer.zero_grad()
                (self.prediction_loss_weight * prediction_loss).backward()
                th.nn.utils.clip_grad_norm_(self.state_predictor.parameters(), self.max_grad_norm)
                self.predictor_optimizer.step()
        
        # Increase update counter
        self._n_updates += gradient_steps
        
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/q_loss", np.mean(losses))
        if prediction_losses:
            self.logger.record("train/prediction_loss", np.mean(prediction_losses))
    
    def predict_next_state(
        self, 
        observation: Union[np.ndarray, Dict[str, np.ndarray]], 
        action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """
        Predict the next state given current observation and action.
        Returns both the predicted state and the prediction (for debugging).
        
        :param observation: Current observation
        :param action: Action to take
        :return: Tuple of (predicted_next_state, prediction_confidence)
        """
        if self.state_predictor is None:
            raise ValueError("State predictor not initialized")
        
        self.state_predictor.eval()
        
        with th.no_grad():
            # Convert to tensors
            if isinstance(observation, dict):
                obs_tensor = th.FloatTensor(observation[next(iter(observation.keys()))]).unsqueeze(0).to(self.device)
            else:
                obs_tensor = th.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            # Handle action conversion more carefully
            if isinstance(action, (int, np.integer)):
                # Ensure the action is within valid range
                action_value = int(action)
                if isinstance(self.action_space, spaces.Discrete):
                    action_value = max(0, min(action_value, self.action_space.n - 1))
                action_tensor = th.LongTensor([action_value]).to(self.device)
            elif isinstance(action, np.ndarray):
                if action.ndim == 0:  # scalar array
                    action_value = int(action.item())
                    if isinstance(self.action_space, spaces.Discrete):
                        action_value = max(0, min(action_value, self.action_space.n - 1))
                    action_tensor = th.LongTensor([action_value]).to(self.device)
                else:
                    action_tensor = th.FloatTensor(action).unsqueeze(0).to(self.device)
            else:
                action_tensor = th.FloatTensor([action]).to(self.device)
            
            # Encode action
            encoded_action = self._encode_action(action_tensor)
            
            # Predict next state
            predicted_next_state = self.state_predictor(obs_tensor, encoded_action)
            
            return predicted_next_state.cpu().numpy().squeeze(), 0.0  # Return dummy confidence for now
    
    def evaluate_prediction_error(
        self, 
        observation: Union[np.ndarray, Dict[str, np.ndarray]], 
        action: Union[int, np.ndarray], 
        actual_next_observation: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> float:
        """
        Evaluate prediction error for a given state-action-next_state tuple.
        
        :param observation: Current observation
        :param action: Action taken
        :param actual_next_observation: Actual next observation
        :return: Prediction error (MSE)
        """
        # Debug: Print action info
        if self.verbose > 1:
            print(f"Debug - Action type: {type(action)}, value: {action}")
            if hasattr(action, 'shape'):
                print(f"Debug - Action shape: {action.shape}")
        
        try:
            predicted_next_state, _ = self.predict_next_state(observation, action)
        except Exception as e:
            print(f"Error in predict_next_state with action {action} (type: {type(action)})")
            print(f"Action space: {self.action_space}")
            raise e
        
        if isinstance(actual_next_observation, dict):
            actual_next_state = actual_next_observation[next(iter(actual_next_observation.keys()))]
        else:
            actual_next_state = actual_next_observation
        
        # Compute MSE
        error = np.mean((predicted_next_state - actual_next_state) ** 2)
        
        if self.eval_mode:
            self.prediction_errors.append(error)
        
        return error
    
    def set_eval_mode(self, eval_mode: bool = True):
        """
        Set evaluation mode to track prediction errors.
        
        :param eval_mode: Whether to enable evaluation mode
        """
        self.eval_mode = eval_mode
        if eval_mode:
            self.prediction_errors = []
    
    def get_prediction_errors(self) -> List[float]:
        """
        Get the list of prediction errors collected during evaluation.
        
        :return: List of prediction errors
        """
        return self.prediction_errors.copy()
    
    def get_high_error_indices(self, threshold_percentile: float = 90) -> List[int]:
        """
        Get indices of states with high prediction errors.
        
        :param threshold_percentile: Percentile threshold for high errors
        :return: List of indices with high prediction errors
        """
        if not self.prediction_errors:
            return []
        
        threshold = np.percentile(self.prediction_errors, threshold_percentile)
        high_error_indices = [i for i, error in enumerate(self.prediction_errors) if error > threshold]
        
        return high_error_indices
    
    def learn(
        self: SelfDQNWithPrediction,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQNWithPrediction",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQNWithPrediction:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
    
    def _excluded_save_params(self) -> List[str]:
        return [*super()._excluded_save_params(), "state_predictor", "predictor_optimizer"]
    
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts, tensors = super()._get_torch_save_params()
        # Add predictor to saved parameters
        state_dicts.extend(["state_predictor", "predictor_optimizer"])
        return state_dicts, tensors 