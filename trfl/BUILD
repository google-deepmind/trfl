# Placeholder for internal Python version compatibility macro.
load(":trfl.bzl", "gen_op_wrapper_py")

licenses(["notice"])  # Apache 2.0 License

package(default_visibility = [":__subpackages__"])

exports_files([
    "LICENSE",
    "setup.py",
])

py_library(
    name = "trfl",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":action_value_ops",
        ":base_ops",
        ":clipping_ops",
        ":discrete_policy_gradient_ops",
        ":dist_value_ops",
        ":dpg_ops",
        ":indexing_ops",
        ":periodic_ops",
        ":pixel_control_ops",
        ":policy_gradient_ops",
        ":retrace_ops",
        ":sequence_ops",
        ":target_update_ops",
        ":value_ops",
        ":vtrace_ops",
    ],
)

py_library(
    name = "action_value_ops",
    srcs = ["action_value_ops.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # visibility removed.
    ],
    deps = [
        ":base_ops",
        ":indexing_ops",
        ":sequence_ops",
        # tensorflow dep,
    ],
)

py_library(
    name = "base_ops",
    srcs = ["base_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        # six dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "clipping_ops",
    srcs = ["clipping_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "discrete_policy_gradient_ops",
    srcs = ["discrete_policy_gradient_ops.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # visibility removed.
    ],
    deps = [
        ":base_ops",
        ":value_ops",
        # six dep,
        # tensorflow dep,
    ],
)

gen_op_wrapper_py(
    name = "gen_distribution_ops",
    srcs = [
        "cc/distribution_ops.cc",
    ],
    out = "gen_distribution_ops.py",
    hdrs = [
        "cc/distribution_ops.h",
    ],
    op_whitelist = [
        "ProjectDistribution",
    ],
    visibility = ["//visibility:private"],
)

py_library(
    name = "distribution_ops",
    srcs = [
        "distribution_ops.py",
    ],
    srcs_version = "PY2AND3",
    visibility = [
        # visibility removed.
    ],
    deps = [
        ":gen_distribution_ops",
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)

py_library(
    name = "dist_value_ops",
    srcs = ["dist_value_ops.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # visibility removed.
    ],
    deps = [
        ":base_ops",
        ":distribution_ops",
        # tensorflow dep,
    ],
)

py_library(
    name = "dpg_ops",
    srcs = ["dpg_ops.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # visibility removed.
    ],
    deps = [
        ":base_ops",
        # tensorflow dep,
    ],
)

py_library(
    name = "indexing_ops",
    srcs = ["indexing_ops.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # visibility removed.
    ],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "periodic_ops",
    srcs = ["periodic_ops.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # visibility removed.
    ],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "pixel_control_ops",
    srcs = ["pixel_control_ops.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # visibility removed.
    ],
    deps = [
        ":action_value_ops",
        ":base_ops",
        # sonnet dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "policy_gradient_ops",
    srcs = ["policy_gradient_ops.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # visibility removed.
    ],
    deps = [
        ":base_ops",
        ":value_ops",
        # six dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "retrace_ops",
    srcs = ["retrace_ops.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # visibility removed.
    ],
    deps = [
        ":base_ops",
        ":indexing_ops",
        ":sequence_ops",
        # six dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "sequence_ops",
    srcs = ["sequence_ops.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # visibility removed.
    ],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "target_update_ops",
    srcs = ["target_update_ops.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # visibility removed.
    ],
    deps = [
        ":periodic_ops",
        # tensorflow dep,
    ],
)

py_library(
    name = "value_ops",
    srcs = ["value_ops.py"],
    srcs_version = "PY2AND3",
    visibility = [
        # visibility removed.
    ],
    deps = [
        ":base_ops",
        ":sequence_ops",
        # tensorflow dep,
    ],
)

py_library(
    name = "vtrace_ops",
    srcs = ["vtrace_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "action_value_ops_test",
    srcs = ["action_value_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":action_value_ops",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "clipping_ops_test",
    srcs = ["clipping_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":clipping_ops",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "discrete_policy_gradient_ops_test",
    srcs = ["discrete_policy_gradient_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":discrete_policy_gradient_ops",
        # absl/testing:parameterized dep,
        # numpy dep,
        # six dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "distribution_ops_test",
    srcs = ["distribution_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":distribution_ops",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)

py_test(
    name = "dist_value_ops_test",
    srcs = ["dist_value_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dist_value_ops",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "dpg_ops_test",
    srcs = ["dpg_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dpg_ops",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "indexing_ops_test",
    srcs = ["indexing_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":indexing_ops",
        # tensorflow dep,
    ],
)

py_test(
    name = "periodic_ops_test",
    srcs = ["periodic_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":periodic_ops",
        # tensorflow dep,
    ],
)

py_test(
    name = "pixel_control_ops_test",
    srcs = ["pixel_control_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pixel_control_ops",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "policy_gradient_ops_test",
    srcs = ["policy_gradient_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":policy_gradient_ops",
        # absl/testing:parameterized dep,
        # numpy dep,
        # six dep,
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)

py_test(
    name = "retrace_ops_test",
    srcs = ["retrace_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":retrace_ops",
        # numpy dep,
        # six dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "sequence_ops_test",
    srcs = ["sequence_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":sequence_ops",
        # numpy dep,
        # six dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "target_update_ops_test",
    srcs = ["target_update_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":target_update_ops",
        # absl/testing:parameterized dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "value_ops_test",
    srcs = ["value_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":value_ops",
        # absl/testing:parameterized dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "vtrace_ops_test",
    srcs = ["vtrace_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":vtrace_ops",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)
