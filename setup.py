"""Setup camrbian library using pip"""

from setuptools import setup

def parse_requirements():
    with open("requirements.txt") as f:
        required = f.read().splitlines()

    return required


setup(
    name='cambrian',
    packages=['cambrian'],
    install_requires=parse_requirements()
)
