# Installing from source

To install TRFL from source, you will need to compile the library using bazel.
You should have installed TensorFlow by following the [TensorFlow installation
instructions](https://www.tensorflow.org/install/).

## Install bazel

Ensure you have a recent version of bazel (>= 0.21.0) and JDK (>= 1.8). If not,
follow [these directions](https://bazel.build/versions/master/docs/install.html).

### (virtualenv TensorFlow installation) Activate virtualenv

If using virtualenv, activate your virtualenv for the rest of the installation,
otherwise skip this step:

```shell
source $VIRTUALENV_PATH/bin/activate # bash, sh, ksh, or zsh
source $VIRTUALENV_PATH/bin/activate.csh  # csh or tcsh
```

### Build and run the installer

First clone the TRFL source code:

```shell
git clone https://github.com/deepmind/trfl
cd trfl
```

Now run the configuration script to fetch the location of you Tensorflow headers:

```shell
./configure.sh
```

Build the `build_pip_pkg` target, which will build all trfl dependencies and the
installation binary:

```shell
bazel build -c opt :build_pip_pkg
```

Finally, use the installation binary to build the wheels and use `pip` to
install them:

```shell
mkdir /tmp/trfl_wheels
./bazel-bin/build_pip_pkg /tmp/trfl_wheels
pip install /tmp/trfl_wheels/*.whl
```
