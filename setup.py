import os
from setuptools import setup, find_packages

_here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(_here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='muri',
    version='0.3',
    description='Scale and Denoise Images Easily',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='David Tsukiyama',
    author_email="davidtsukiyama1@gmail.com",
    packages=find_packages(exclude=['tests*']),
    include_package_data=True,
    license='MIT',
    url='https://github.com/dtsukiyama/muri',
    install_requires=[
        'chainer',
        'numpy',
        'Pillow',
        'pytest',
        'six'
    ]
)
