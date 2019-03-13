#!/bin/bash

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

set -e

if [ "$#" -ne 2 ]; then
    echo "usage: %prog% path/to/op_lib.so op1,op2,..."
    exit 1
fi

cat <<EOF
import tensorflow as tf
_op_lib = tf.load_op_library(tf.resource_loader.get_path_to_datafile("$1"))
EOF

for name in $(echo $2 | tr "," "\n")
do
    snake_name=$(python -c "import stringcase; import sys; print(stringcase.snakecase(sys.argv[1]))" $name)
    echo "$snake_name = _op_lib.$snake_name"
done

echo "del _op_lib, tf"
