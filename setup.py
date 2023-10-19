import pathlib

import pkg_resources
import setuptools

here = pathlib.Path(__file__).parent.absolute()

with pathlib.Path(here / "requirements.txt").open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]

with pathlib.Path(here / "requirements_tests.txt").open() as requirements_txt:
    extra_requires = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]

DESCRIPTION = "PyTorch Lightning Experiment Logger"

try:
    with open(here / "README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setuptools.setup(
    name="sagemaker-experiments-logger",
    version="1.0.0",
    author="Tobias Senst",
    author_email="tobias.senst@googlemail.com",
    description="PyTorch Lightning Experiment Logger",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="ssh://git@github.com:tsenst/lightning-experiments-logger.git",
    install_requires=install_requires,
    extras_require={
        "dev": ["black", "flake8", "isort", "mypy", "pylint", "types-PyYAML"],
        "tests": extra_requires,
    },
    packages=setuptools.find_packages(exclude="tests"),
    keywords="pytorch-lightning, AWS SageMaker, machine learning",
    license="Apache 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: APACHE 2.0",
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
