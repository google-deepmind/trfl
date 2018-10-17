# Copyright 2018 The trfl Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Setup for pip package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['six', 'absl-py', 'numpy', 'dm-sonnet']
EXTRA_PACKAGES = {
    'tensorflow': ['tensorflow>=1.8.0', 'tensorflow-probability>=0.4.0'],
    'tensorflow with gpu': ['tensorflow-gpu>=1.8.0',
                            'tensorflow-probability-gpu>=0.4.0'],
}


def trfl_test_suite():
  test_loader = unittest.TestLoader()
  test_suite = test_loader.discover('trfl', pattern='*_test.py')
  return test_suite


setup(
    name='trfl',
    version='1.0',
    description=('trfl is a library of building blocks for '
                 'reinforcement learning algorithms.'),
    long_description='',
    url='http://www.github.com/deepmind/trfl/',
    author='DeepMind',
    author_email='trfl-steering@google.com',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    zip_safe=False,
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
    ],
    keywords='trfl truffle tensorflow tensor machine reinforcement learning',
    test_suite='setup.trfl_test_suite',
)
