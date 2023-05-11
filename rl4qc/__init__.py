from gym.envs.registration import register

register(
    id="rl4qc/CircuitDesigner-v0",
    entry_point="rl4qc.env:CircuitDesigner",
)