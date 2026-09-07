from setuptools import setup, find_packages

setup(
    name="t2v_turbo",
    version="0.1",
    packages=find_packages(),  # will pick up reward_fn, src, etc.
    install_requires=[
        # any dependencies, e.g.
        # "torch>=1.12.0",
    ],
)