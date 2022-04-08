from setuptools import setup

setup(
    name='QGP-evolution-predictor',
    version='1.0',
    py_modules=['main', 'utils', 'model'],
    setup_requires=['wheel'],
    install_requires=[
        'Click',
        'torch',
        'numpy>=1.7',
        'matplotlib'
    ],
    entry_points='''
        [console_scripts]
        qgpepr=main:cli
    '''
)