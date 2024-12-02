from setuptools import setup, find_packages


setup(
    name='llm-genjax',
    version="0.0.1",
    packages=[package for package in find_packages() if package.startswith('llm_genjax')],
    install_requires=[
        "jax",
        "flax",
        "anthropic",
        "python-dotenv",
        "matplotlib",
        "pandas",
        "seaborn",
        "dataclasses-json",
        "shutup"
    ],
    description='',
    author='Nathan Cloos'
)
