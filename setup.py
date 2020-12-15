from setuptools import setup, find_packages

setup(
    name="unipoint",
    version="1.0",
    packages=find_packages(),
    install_requires=['torch', 'numpy', 'matplotlib', 'tqdm', 'sklearn']
)