import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard.writer import SummaryWriter
from algorithm.logging import write_hyperparameters

class EvaluationCallback(BaseCallback):
  """ Callback for evaluating an agent.
  :param model: The model to be evaluated^
  :param eval_envs: A dict containing environments for testing the current model.
  :param stop_on_reward: Whether to use early stopping. Defaults to True
  :param reward_threshold: The reward threshold to stop at."""
  def __init__(self, model: BaseAlgorithm, eval_envs: dict, stop_on_reward:float=None, record_video:bool=True, run_test:bool=True):
    super(EvaluationCallback, self).__init__(); self.model = model; self.writer: SummaryWriter = self.model.writer
    self.eval_envs = eval_envs; self.record_video = record_video; self.run_test = run_test
    self.stop_on_reward = lambda r: (stop_on_reward is not None and r >= stop_on_reward) or not self.model.continue_training
    if stop_on_reward is not None: print(f"Stopping at {stop_on_reward}"); assert run_test, f"Can't stop on reward {stop_on_reward} without running test episodes"
    if record_video: assert run_test, f"Can't record video without running test episodes"

  def _on_training_start(self):  self.evaluate()

  def _on_rollout_end(self) -> None:
    if self.writer == None: return 
    # Uncomment for early stopping based on 100-mean training return
    mean_return = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
    if self.stop_on_reward(mean_return): self.model.continue_training = False
    if self.model.should_eval(): self.evaluate()

  def _on_step(self) -> bool: 
    """ Write timesteps to info & stop on reward threshold"""
    [info['episode'].update({'t': self.model.num_timesteps}) for info in self.locals['infos'] if info.get('episode')]
    return self.model.continue_training

  def _on_training_end(self) -> None: # No Early Stopping->Unkown, not reached (continue=True)->Failure, reached (stopped)->Success
    if self.writer == None: return 
    status = 'STATUS_UNKNOWN' if not self.stop_on_reward else 'STATUS_FAILURE' if self.model.continue_training else 'STATUS_SUCCESS'
    metrics = self.evaluate(); write_hyperparameters(self.model, list(metrics.keys()), status)

  def evaluate(self):
    """Run evaluation & write hyperparameters, results & video to tensorboard. Args:
        write_hp: Bool flag to use basic method for writing hyperparams for current evaluation, defaults to False
    Returns: metrics: A dict of evaluation metrics, can be used to write custom hparams """ 
    step = self.model.num_timesteps
    if not self.writer: return []
    metrics = {k:v for label, env in self.eval_envs.items() for k, v in self.run_eval(env, label, step).items()}
    [self.writer.add_scalar(key, value, step) for key, value in metrics.items()]; self.writer.flush()  
    return metrics

  def run_eval(self, env, label: str, step: int):
    metrics = {}
    if self.run_test: 
      deterministic = False  # not env.get_attr('spec')[0].nondeterministic
      n_eval_episodes = 1 if not deterministic else 100
      n_envs = env.num_envs; episode_rewards = []; episode_counts = np.zeros(n_envs, dtype="int")
      episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
      observations = env.reset(); states = None; episode_starts = np.ones((env.num_envs,), dtype=bool)
      while (episode_counts < episode_count_targets).any():
        actions, states = self.model.predict(observations, state=states, episode_start=episode_starts, deterministic=deterministic)
        new_observations, _, dones, infos = env.step(actions)
        for i in range(n_envs):
          if episode_counts[i] < episode_count_targets[i]:
            episode_starts[i] = dones[i]
            if dones[i] and "episode" in infos[i].keys():
              episode_rewards.append(infos[i]["episode"]["r"]); episode_counts[i] += 1
        observations = new_observations
      metrics[f"rewards/{label}"] = np.mean(episode_rewards) #np.std(episode_rewards)
      if self.record_video: env.envs[0].write_video(self.writer, label, step)
    self.writer.flush()
    return metrics
