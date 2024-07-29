from gymnasium.envs.registration import register, registry

def register_envs(): 
  if "CircuitDesigner-v0" in registry: return
  register(id="CircuitDesigner-v0", entry_point="qcd_gym.env:CircuitDesigner")