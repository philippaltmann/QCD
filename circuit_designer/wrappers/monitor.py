import time; import numpy as np
import gymnasium as gym
from gymnasium.core import ActType, ObsType; 
from typing import SupportsFloat

class Monitor(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
  """ A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.
  :param env: The environment """
  def __init__( self, env: gym.Env, record_video=False, rescale=True, discrete=False):
    super().__init__(env=env); self.t_start = time.time(); self.rescale = rescale
    # if self.rescale:
    #   _l = self.action_space.low; _h = self.action_space.high
    #   self._rescale = lambda action: np.clip((_l+ 0.5 * (action + 1.0)) * (_h - _l), _l, _h)
    self.record_video = record_video; self._frame_buffer = []; self.discrete = discrete
    if discrete: self.action_space = gym.spaces.Discrete(1334*self.qubits+1)
    self.states: list = [np.ndarray]; self.actions:list = [np.ndarray]; self.rewards: list[float] = []
    self._history = lambda: {key: getattr(self,key).copy() for key in ['states','actions','rewards']}
    self._episode_returns: list[float] = []; self._termination_reasons: list[str] = []
    self._episode_lengths: list[int] = []; self._episode_times: list[float] = []; 
    self._total_steps = 0; self.needs_reset = True; self.rescale = rescale
    self._ep_depths, self._ep_qubits, self._ep_operations = [], [], [] # Q-Metrics

  def reset(self, **kwargs) -> tuple[ObsType, dict]:
    """ Calls the Gym environment reset. 
    :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
    :return: the first observation of the environment """
    self.needs_reset = False
    state, info = self.env.reset(**kwargs)
    self.rewards = []; self.states=[state]; self.actions = []
    if self.record_video: self._frame_buffer.append(self.render())
    return state, info

  def _discrete(self, action: ActType): 
    """     Discrete Mapping: 
    [0; 36*max_qubis[ -> Z-Rotation for qubit 1..max_qubit
    [36*max_qubis ..36*max_qubis+1296*max_qubits[]Phased-X \w parameters [0..35][0..35][0..n_qubits]
    [36*max_qubis+1296*max_qubis .. 36*max_qubis+1296*max_qubis+max_qubis*2[
    """
    print(action)
    o_border = [36*self.qubits, 1332*self.qubits, 1334*self.qubits, 1334*self.qubits+1]
    operation = 0 if action < o_border[0] else 1 if action < o_border[1] else 2 if action < o_border[2] else 3
    print(operation)
    w_action = action - [0, *o_border][operation]
    wire = w_action % self.qubits if operation < 3 else 0
    phi = 0
    theta = 0

    _discrete_action = [operation, wire, phi, theta]
    return _discrete_action

  def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
    """ Step the environment with the given action
    :param action: the action
    :return: observation, reward, terminated, truncated, information """
    if self.needs_reset: raise RuntimeError("Tried to step environment that needs reset")
    # if self.rescale: action = self._rescale(action)
    # if self.discrete: action = self._discrete(action)
    state, reward, terminated, truncated, info = self.env.step(action)
    self.states.append(state); self.actions.append(action); self.rewards.append(float(reward))
    if self.record_video: self._frame_buffer.append(self.render())
    if terminated or truncated:
      self.needs_reset = True; ep_rew = sum(self.rewards); ep_len = len(self.rewards)
      ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6), 'history': self._history()}
      self._episode_returns.append(ep_rew); 
      self._termination_reasons.append(info.pop('termination_reason'))
      self._episode_lengths.append(ep_len); self._episode_times.append(time.time() - self.t_start)

      ep_info['d'] = info["resources"].depth      # ['depth']
      ep_info['q'] = info["resources"].num_wires  # ['num_used_wires']
      ep_info['o'] = info["resources"].num_gates  # ['num_operations']
      info["episode"] = ep_info

    self._total_steps += 1
    return state, reward, terminated, truncated, info
  
  def get_video(self, reset=True):
    frame_buffer = self._frame_buffer.copy()
    if reset: self._frame_buffer = []
    return np.array(frame_buffer)
  
  def write_video(self, writer, label, step):
    """Adds current videobuffer to tensorboard"""
    frame_buffer =  self.get_video()
    if self.render_mode  == 'text': writer.add_text(label, frame_buffer[-2], step)
    elif self.render_mode  == 'image': assert False, 'Not implemented'
  
  @property
  def total_steps(self) -> int: return self._total_steps

  @property
  def episode_returns(self) -> list[float]: return self._episode_returns

  @property
  def termination_reasons(self) -> list[str]: return self._termination_reasons

  @property
  def episode_lengths(self) -> list[int]: return self._episode_lengths

  @property
  def episode_times(self) -> list[float]: return self._episode_times

