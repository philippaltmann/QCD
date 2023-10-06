from gymnasium.envs.registration import register
from .utils import factory

# def register_envs():
register(id="CircuitDesigner-v0", entry_point="circuit_designer.env:CircuitDesigner")