from setuptools import setup, find_packages


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='muri',
    version='0.1',
    description='Scale and Denoise Images Easily',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='David Tsukiyama',
    author_email="davidtsukiyama1@gmail.com",
    modules=['muda','yare','kantan','maji'],
    packages=find_packages(),
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
