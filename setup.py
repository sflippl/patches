#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on DATE Mon May 20 12:20:33 2019

@author: sflippl
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PKGNAME",
    version="VERSIONNR",
    author="Samuel Lippl",
    author_email="sfc.lippl@gmail.com",
    description="DESCRIPTION",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="URL",
    packages=setuptools.find_packages(),
    include_package_data = True,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
)
