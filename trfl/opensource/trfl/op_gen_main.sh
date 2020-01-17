#!/bin/bash

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

set -e

if [ "$#" -ne 2 ]; then
    echo "usage: %prog% path/to/op_lib.so op1,op2,..."
    exit 1
fi

cat <<EOF
import os.path
import tensorflow as tf
_op_lib = tf.load_op_library(os.path.join(os.path.dirname(__file__), "$1"))
EOF

for name in $(echo $2 | tr "," "\n")
do
    snake_name=`echo -n $name | python -c "import re, sys; [s] = sys.stdin; sys.stdout.write(re.sub(r'(.)([A-Z])', r'\1_\2', s).lower())"`
    echo "$snake_name = _op_lib.$snake_name"
done

echo "del _op_lib, tf"
