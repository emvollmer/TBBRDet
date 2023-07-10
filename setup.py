from setuptools import setup, find_packages
import subprocess

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setup(
    name='TBBRDet',
    version='0.1.0',
    author='James Kahn',
    author_email='kahn.jms@gmail.com',
    description='Thermal Bridges on Building Rooftops Detection',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/emvollmer/TBBRDet.git',
    license='BSD-3-Clause',
    classifiers=[
        'Intended Audience :: Information Technology',
        'Intended Audience :: System Administrators',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(),
    install_requires=requirements
)
