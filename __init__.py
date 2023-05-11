from environment import CircuitDesigner
from gym.envs.registration import register

register(
    id="CircuitDesigner",
    entry_point="RL4QC:CircuitDesigner",
)