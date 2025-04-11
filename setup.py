from setuptools import setup, find_packages

setup(
    name="chess_rl",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pygame",
        "python-chess",
        "torch",
        "numpy",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A chess AI using reinforcement learning",
)