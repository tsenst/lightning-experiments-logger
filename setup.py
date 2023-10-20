import pathlib

import pkg_resources
import setuptools

here = pathlib.Path(__file__).parent.absolute()

DESCRIPTION = "PyTorch Lightning Experiment Logger"

try:
    with open(here / "README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setuptools.setup(
    name="sagemaker-experiments-logger",
    version="0.1.1",
    author="Tobias Senst",
    author_email="tobias.senst@googlemail.com",
    description="PyTorch Lightning Experiment Logger",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tsenst/lightning-experiments-logger",
    install_requires=[
        "pytorch-lightning>=2.0.0",
        "sagemaker>=2.190.0",
    ],
    extras_require={
        "dev": ["black", "flake8", "isort", "mypy", "pylint", "types-PyYAML"],
        "tests": ["moto==4.2.5", "pytest", "pytest-mock", "pytest-cov"]
    },
    packages=setuptools.find_packages(exclude="tests"),
    keywords="pytorch-lightning, AWS SageMaker, machine learning",
    license="Apache 2.0",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
)
