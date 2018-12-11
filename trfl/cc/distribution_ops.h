/* Copyright 2018 The trfl Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TRFL_CC_DISTRIBUTION_OPS_H_
#define TRFL_CC_DISTRIBUTION_OPS_H_

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base/thread_annotations.h"
#include "third_party/absl/strings/string_view.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "util/gtl/array_slice.h"

namespace deepmind {
namespace rl {

class ProjectDistribution : public tensorflow::OpKernel {
 public:
  // WARNING: please do not use this C++ op directly and rather rely
  // on python wrappers provided in distribution_ops.py.
  // The interface is not fixed and may change in the future.
  //
  // Depending on the provided method_id, the op executes one of several
  // operations on categorical distributions. Please refer to the documentation
  // of `Compute` for more information.
  explicit ProjectDistribution(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override;
};


}  // namespace rl
}  // namespace deepmind

#endif  // TRFL_CC_DISTRIBUTION_OPS_H_
