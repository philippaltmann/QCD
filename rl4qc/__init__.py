from gymnasium.envs.registration import register

register(
    id="CircuitDesigner-v0",
    entry_point="rl4qc.env:CircuitDesigner",
)