"""Generate wrappers for custom TF ops."""

def gen_op_wrapper_py(
        name,
        out,
        op_whitelist,
        srcs,
        hdrs = [],
        visibility = None):
    """Generates a Python library target wrapping the given ops."""

    # NOTE(slebedev): _ is neded to avoid collision with the py_library.
    op_lib_so_name = "_" + name + ".so"
    native.cc_binary(
        # NOTE(slebedev): has to have the shared library suffix, otherwise
        # Bazel complains about linkshared = 1.
        name = op_lib_so_name,
        srcs = srcs + hdrs,
        visibility = ["//visibility:private"],
        deps = [
            "@local_config_tf//:libtensorflow_framework",
            "@local_config_tf//:tf_header_lib",
        ],
        linkshared = 1,
        copts = ["-D_GLIBCXX_USE_CXX11_ABI=0"],
    )

    native.genrule(
        name = name + "_pygenrule",
        outs = [out],
        visibility = visibility,
        cmd = ("$(location op_gen_main.sh) " + op_lib_so_name + " " +
               ",".join(op_whitelist) + " > $@"),
        tools = ["op_gen_main.sh"],
    )

    native.py_library(
        name = name,
        srcs = [out],
        srcs_version = "PY2AND3",
        visibility = ["//visibility:private"],
        data = [
            ":" + op_lib_so_name,
        ],
    )
