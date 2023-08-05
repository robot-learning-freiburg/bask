"""Install script for setuptools."""

from os import path

import setuptools

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="bask",
    version="0.1.0",
    install_requires=[],
    packages=setuptools.find_packages(),
    python_requires=">=3.7"
    )
