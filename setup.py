import os
from setuptools import setup, find_packages


def read(fname):
    """Reads a file's contents as a string.
    Args:
        fname: Filename.
    Returns:
        File's contents.
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


VERSION = "0.1"
BASE_URL = "https://github.com/imgeorgiev/ddp"
INSTALL_REQUIRES = [
    "numpy>=1.19",
    "sympy>=1.10",
]

setup(
    name="ddp",
    version=VERSION,
    description="Finite horizon Discrete-time Differential Dynamic Programming(DDP)",
    long_description=read("README.md"),
    author="Ignat Georgiev",
    author_email="ignat@imgeorgiev.com",
    license="MIT",
    url=BASE_URL,
    packages=find_packages(),
    zip_safe=True,
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education" "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
