import time; import numpy as np
import gymnasium as gym
from gymnasium.core import ActType, ObsType; 
from typing import SupportsFloat

class Monitor(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
  """ A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.
  :param env: The environment """
  def __init__( self, env: gym.Env, record_video=False):
    super().__init__(env=env); self.t_start = time.time(); 
    self.record_video = record_video; self._frame_buffer = []; 
    self.states: list = [np.ndarray]; self.actions:list = [np.ndarray]; self.rewards: list[float] = []
    self._history = lambda: {key: getattr(self,key).copy() for key in ['states','actions','rewards']}
    self._episode_returns: list[float] = []; self._termination_reasons: list[str] = []
    self._episode_lengths: list[int] = []; self._episode_times: list[float] = []; 
    self._total_steps = 0; self.needs_reset = True
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

  def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
    """ Step the environment with the given action
    :param action: the action
    :return: observation, reward, terminated, truncated, information """
    if self.needs_reset: raise RuntimeError("Tried to step environment that needs reset")
    state, reward, terminated, truncated, info = self.env.step(action)
    self.states.append(state); self.actions.append(action); self.rewards.append(float(reward))
    if self.record_video: self._frame_buffer.append(self.render())
    if terminated or truncated:
      self.needs_reset = True; ep_rew = sum(self.rewards); ep_len = len(self.rewards)
      ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6), 'history': self._history()}
      self._episode_returns.append(ep_rew); 
      self._termination_reasons.append(info.pop('termination_reason'))
      self._episode_lengths.append(ep_len); self._episode_times.append(time.time() - self.t_start)

      ep_info['d'] = info["depth"]
      ep_info['o'] = info["operations"]
      ep_info['q'] = info["used_wires"]
      ep_info['m'] = info["metric"]
      ep_info['c'] = info["cost"]
      info["episode"] = ep_info

    self._total_steps += 1
    return state, reward, terminated, truncated, info
  
  def get_video(self, reset=True):
    frame_buffer = self._frame_buffer.copy()
    if reset: self._frame_buffer = []
    return np.array(frame_buffer)
  
  def write_video(self, writer, label, step):
    """Adds current videobuffer to tensorboard"""
    if self.render_mode  == 'text': writer.add_text(label, str(self.env.render()), step)
  
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

