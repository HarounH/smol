# pip install -e .
from setuptools import setup, find_packages

setup(
    name="smol",
    version="0.1.0",
    description="Working project",
    packages=find_packages(include=["smol", "smol.*"]),
    python_requires=">=3.8",
    install_requires=[
        "imageio>=2.19",
        "numpy>=1.21",
        "Pillow>=9.2",
        "mss>=9.0",
    ],
)
