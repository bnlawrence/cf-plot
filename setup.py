from setuptools import find_packages, setup
import os
import fnmatch
import sys
import importlib
import subprocess


def find_package_data_files(directory):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, "*"):
                filename = os.path.join(root, basename)
                yield filename.replace("cfplot/", "", 1)


package_data = [
    f for f in find_package_data_files("cfplot/colour/colourmaps")
]

setup(
    name="cf-plot",
    version="3.4.0",
    author="Andy Heaps",
    author_email="andy.heaps@ncas.ac.uk",
    maintainer="Sadie Bartholomew",
    maintainer_email="sadie.bartholomew@ncas.ac.uk",
    packages=find_packages(),
    package_dir={"cfplot": "cfplot"},
    package_data={"cfplot": package_data},
    include_package_data=True,
    install_requires=[
        # "matplotlib >=3.1.0",  # already a requirement for Cartopy
        "cf-python >= 3.17.0",
        # "scipy >= 1.4.0",  # already a requirement for cf-python
        "cartopy >= 0.22.0",
    ],
    url="https://ncas-cms.github.io/cf-plot/",
    license="LICENSE.txt",
    description="Code-light plotting for earth science and aligned research",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
