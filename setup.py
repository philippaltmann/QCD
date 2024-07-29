from pathlib import Path
from setuptools import setup, find_packages

train = [ "torch==2.0.1", "stable_baselines3>=2.0.0", "tqdm>=4.65.0"]
plot = ["plotly>=5.0", "tensorboard>=2.0"]

setup(
  name="qcd-gym", version="0.2.0",
  description="Quantum Circuit Designer: A gymnasium-based set of environments for benchmarking reinforcement learning for quantum circuit design.",
  url="https://github.com/philippaltmann/qcd", author_email="philipp@hyphi.co", license="MIT",
  keywords="benchmark reinforcement-learning quantum-computing gymnasium circuit-design",
  long_description=(Path(__file__).parent / "README.md").read_text(), long_description_content_type="text/markdown",
  packages=find_packages(include=['qcd_gym','qcd_gym.wrappers']),
  install_requires=[ "gymnasium==0.29", "qiskit==1.0.2" ],
  extras_require = { 
    "tests": [ "pytest", "black"], 
    "train": train, "plot": plot,
    "all": train + plot
  },
  python_requires=">=3.8",
  entry_points={ "gymnasium.envs": ["__root__ = qcd_gym.__init__:register_envs"] }
)
