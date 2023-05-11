from environment import CircuitDesigner
from gym.envs.registration import register

register(
    id="RL4QC/CircuitDesigner",
    entry_point="RL4QC:CircuitDesigner",
)