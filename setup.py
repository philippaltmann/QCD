from pathlib import Path
from setuptools import setup, find_packages

setup(
  name="circuit_designer", version="0.1",
  description="Challenges for Quantum Reinforcement Learning: A gymnasium-based set of environments for benchmarking quantum circuit design.",
  url="https://github.com/philippaltmann/qrl", author_email="philipp@hyphi.co", license="MIT",
  keywords="reinforcement-learning quantum-computing benchmark gymnasium cirquit-design",
  long_description=(Path(__file__).parent / "README.md").read_text(), long_description_content_type="text/markdown",
  packages=[package for package in find_packages() if package.startswith("circuit_designer")],
  install_requires=[
    "gymnasium>=0.28.1",
    "pennylane==0.31.1",
    "numpy>=1.20",
    "scipy>=1.10",
  ],
  python_requires=">=3.8",
  classifiers=[
      "Programming Language :: Python :: 3",
      "Programming Language :: Python :: 3.8",
      "Programming Language :: Python :: 3.9",
      "Programming Language :: Python :: 3.10",
      "Programming Language :: Python :: 3.11",
  ],
  entry_points={
      "gymnasium.envs": ["__root__ = circuit_designer.__init__:register_envs"]
  }
)