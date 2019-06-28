from setuptools import setup, find_packages

setup(
    name='muri',
    version='0.1',
    description='Scale Images',
    author='David Tsukiyama',
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
