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

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase

REQUIRED_PACKAGES = ['six', 'absl-py']

EXTRA_PACKAGES = {
    'tensorflow': ['tensorflow>=1.0.1'],
    'tensorflow with gpu': ['tensorflow-gpu>=1.0.1']
}

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
    cmdclass={
        'install': InstallCommandBase,
    },
    license='Apache 2.0',
    keywords='trfl truffle tensorflow tensor machine reinforcement learning',
)
