# Tests for `trfl` ops using TF2.
# Placeholder for internal Python version compatibility macro.

# Apache 2.0 License
licenses(["notice"])

package(default_visibility = ["//visibility:public"])

py_test(
    name = "value_ops_test",
    srcs = ["value_ops_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # tensorflow dep,
        # tree dep,
        "//trfl:value_ops",
    ],
)

py_test(
    name = "action_value_ops_test",
    srcs = ["action_value_ops_test.py"],
    deps = [
        # tensorflow dep,
        "//trfl:action_value_ops",
    ],
)

py_test(
    name = "clipping_ops_test",
    srcs = ["clipping_ops_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//trfl:clipping_ops",
    ],
)

py_test(
    name = "policy_ops_test",
    srcs = ["policy_ops_test.py"],
    deps = [
        # tensorflow dep,
        "//trfl:policy_ops",
    ],
)
