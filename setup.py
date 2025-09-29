# pip install -e .
from setuptools import setup, find_packages

setup(
    name="smol",
    version="0.1.0",
    description="Working project",
    packages=find_packages(include=["smol", "smol.*"]),
    python_requires=">=3.8",
    install_requires=[],  # list runtime deps here if you have any
)


