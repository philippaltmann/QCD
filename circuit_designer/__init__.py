from gymnasium.envs.registration import register

# def register_envs():
register(id="CircuitDesigner-v0", entry_point="circuit_designer.env:CircuitDesigner")