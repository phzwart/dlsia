#!/usr/bin/env python


from os import path

"""The setup script."""

from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent


with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

here = path.abspath(path.dirname(__file__))



with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Petrus H. Zwart, Eric J. Roberts",
    author_email='PHZwart@lbl.gov, EJroberts@lbl.gov',
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
    ],
    description="Deep Learning for Scientif Image Analysis",
    install_requires=requirements,
    license="BSD License",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords='dlsia',
    name='dlsia',
    packages=find_packages(include=['dlsia', 'dlsia.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/phzwart/dlsia/',
    version='0.3.1',
    zip_safe=False,
)
