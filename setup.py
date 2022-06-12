
from setuptools import setup

import os

long_desc="AttentionLSTM",

fpath = os.path.join(os.getcwd(), "readme.md")
if os.path.exists(fpath):
    with open(fpath, "r") as fd:
        long_desc = fd.read()

setup(

    name='AttentionLSTM',

    version="0.1",

    description='Combining LSTM with attention',
    long_description=long_desc,
    long_description_content_type="text/markdown",

    url='https://github.com/AtrCheema/AttentionLSTM',

    author='Ather Abbas',
    author_email='ather_abbas786@yahoo.com',

    classifiers=[
        'Development Status :: 4 - Beta',

        'Natural Language :: English',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',

        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

    packages=['atten_lstm'],

    install_requires=[
        'tensorflow',
    ]
)