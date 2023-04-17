import os
from setuptools import find_packages, setup

setup(
    name="ReDMan",
    packages=[package for package in find_packages() if package.startswith("ReDMan")],
    package_data={"ReDMan": ["py.typed", "version.txt"]},
    install_requires=[
        "psutil",
        "joblib",
        "tensorboard",
        "pyyaml",
        "matplotlib",
        "pandas",
        "tensorboardX",
        "gym",
        "wandb"
        # we recommend use conda install scipy and mpi4py
    ],
    description="Pytorch Version of Safe Reinforcement Learning Algorithm for Dexterous Manipulation",
    author="OmniSafeAI",
    url="https://github.com/OmniSafeAI/ReDMan.git",
    author_email="None",
    keywords="Safe Reinforcement Learning",
    license="Apache License 2",
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)