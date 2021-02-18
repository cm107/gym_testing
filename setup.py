from setuptools import setup, find_packages
import gym_testing

packages = find_packages(
        where='.',
        include=['gym_testing*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pyclay-gym_testing',
    version=gym_testing.__version__,
    description='Gym testing',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cm107/gym_testing",
    author='Clayton Mork',
    author_email='mork.clayton3@gmail.com',
    license='MIT License',
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'torch',
        'torchvision',
        'gym',
        'pylint==2.4.4',
        'pyclay-common_utils @ https://github.com/cm107/common_utils/archive/master.zip'
    ],
    python_requires='>=3.7'
)