from setuptools import find_namespace_packages, setup

setup(
    name="microsat_rl",
    version="0.1.0",
    description="CS 295 Reinforcement Learning: Microsatellite Attitude Control",
    author="James-Andrew Sarmiento",
    packages=find_namespace_packages(include=["src*"]),
    python_requires="~=3.6.9",
    install_requires=[
        "gym~=0.18.3",
        "gym-unity~=0.19.0",
        "mlagents~=0.19.0",
        "mlagents-envs~=0.19.0,
        "torch~=1.8.1",
        "tensorflow~=1.14.0",
        "mpi4py~=3.0.3"
        "mujoco-py~=1.50.1.68",
        "numpy~=1.18.5"
    ],
)