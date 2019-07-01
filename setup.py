from setuptools import setup, find_packages

setup(
    name='muri',
    version='0.1',
    description='Scale and Denoise Images Easily',
    long_description=open('README.md').read(),
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
