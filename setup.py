from importlib.metadata import entry_points
from setuptools import setup

setup(
    name='QGP-evolution-predictor',
    version='1.0',
    py_modules=['qgpepr'],
    install_requires=[
        'Click',
        'torch',
        'numpy'
    ],
    entry_points='''
        [console_scripts]
        qgpepr=qgpepr:cli
    '''
)