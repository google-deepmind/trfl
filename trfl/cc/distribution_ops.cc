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

#include "distribution_ops.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

namespace deepmind {
namespace rl {

namespace {

// Executes a static shape inference for a particular dimension.
// 1. Fills out shape information if one of the tensors has an unknown shape.
// 2. Broadcasts shape information if one of the tensors has shape 1.
// 3. Checks that dimensions are otherwise compatible.
// Args:
//   c: the shape inference context;
//   d0, d1: sizes of a given dimension of two input tensors;
//   out: the resulting dimension.
// Returns:
//   error code if dimensions are incompatible.
tensorflow::Status MergeWithBroadcastStatic(
    tensorflow::shape_inference::InferenceContext* c,
    tensorflow::shape_inference::DimensionHandle d0,
    tensorflow::shape_inference::DimensionHandle d1,
    tensorflow::shape_inference::DimensionHandle* out) {
  if (!c->ValueKnown(d1) || c->Value(d1) == 1) {
    *out = d0;
    return tensorflow::Status::OK();
  } else if (!c->ValueKnown(d0) || c->Value(d0) == 1) {
    *out = d1;
    return tensorflow::Status::OK();
  } else if (c->Value(d0) == c->Value(d1)) {
    *out = d0;
    return tensorflow::Status::OK();
  } else {
    return tensorflow::errors::InvalidArgument("Incompatible dimensions",
         c->Value(d0), " and ", c->Value(d1));
  }
}

// Executes a dynamic shape inference for a particular dimension.
// The function is consistent with MergeWithBroadcastStatic except
// that all dimensions are known at run-time.
// 1. Broadcasts shape information if one of the tensors has shape 1.
// 2. Checks that dimensions are otherwise compatible.
// Args:
//   d0, d1: sizes of a given dimension of two input tensors;
//   out: the resulting dimension.
// Returns:
//   error code if dimensions are incompatible.
tensorflow::Status MergeWithBroadcastDynamic(
    int64_t d0,
    int64_t d1,
    int64_t* out) {
  if ((d0 == d1) || (d1 == 1)) {
    *out = d0;
    return tensorflow::Status::OK();
  } else if (d0 == 1) {
    *out = d1;
    return tensorflow::Status::OK();
  } else {
    return tensorflow::errors::InvalidArgument("Incompatible dimensions",
        d0, " and ", d1);
  }
}

// Performs static shape inference for ProjectDistribution op.
// Inputs tensors are: support_old, weights, support_new.
// Input tensors have to satisfy the following requirements:
// 1. All ranks have to be known.
// 2. Ranks of all supports have to be 1 or equal to the rank of weights.
// 3. Broadcasting happens along all dimensions except the last one. This means
//    that sizes of any given dimension (except the last) must be equal or
//    be of the size 1 or be unknown.
// 4. support_old and weights have to have an equal last dimension which
//    represents the number of bins or one of the values has to be unknown.
// 5. support_new can have the last dimension of any size.
tensorflow::Status ProjectDistributionShapeFn(
  tensorflow::shape_inference::InferenceContext* c) {
  tensorflow::shape_inference::ShapeHandle shape_s1 = c->input(0);
  tensorflow::shape_inference::ShapeHandle shape_p = c->input(1);
  tensorflow::shape_inference::ShapeHandle shape_s2 = c->input(2);
  tensorflow::shape_inference::ShapeHandle shape_m = c->input(3);
  // Make sure ranks are known.
  if (!c->RankKnown(shape_s1)) {
    return tensorflow::errors::InvalidArgument(
        "support tensor must have a known rank.");
  }
  if (!c->RankKnown(shape_p)) {
    return tensorflow::errors::InvalidArgument(
        "weights tensor must have a known rank.");
  }
  if (!c->RankKnown(shape_s2)) {
    return tensorflow::errors::InvalidArgument(
        "new_support tensor must have a known rank.");
  }
  if (!c->RankKnown(shape_m)) {
    return tensorflow::errors::InvalidArgument(
        "method tensor must have a known rank.");
  }

  // Make sure ranks are consistent.
  const int32_t rank_s1 = c->Rank(shape_s1);
  const int32_t rank_s2 = c->Rank(shape_s2);
  const int32_t rank_p = c->Rank(shape_p);
  const int32_t rank_m = c->Rank(shape_m);
  bool eq_rank_s1_and_p = (rank_s1 == rank_p);
  bool eq_rank_s2_and_p = (rank_s2 == rank_p);
  if (!(rank_s1 == 1 || eq_rank_s1_and_p)) {
    return tensorflow::errors::InvalidArgument(
        "support must be a vector or have the same rank as weights");
  }
  if (!(rank_s2 == 1 || eq_rank_s2_and_p)) {
    return tensorflow::errors::InvalidArgument(
        "new_support must be a vector or have the same rank as weights");
  }
  if (!(rank_m == 0)) {
    return tensorflow::errors::InvalidArgument(
        "method tensor must have a rank of 0");
  }

  // Make sure the number of bins is consistent.
  auto bins_p = c->Dim(shape_p, rank_p - 1);
  auto bins_s1 = c->Dim(shape_s1, rank_s1 - 1);
  if (c->ValueKnown(bins_p) && c->ValueKnown(bins_s1) &&
      (c->Value(bins_p) != c->Value(bins_s1))) {
    return tensorflow::errors::InvalidArgument(
        "support and weights tensors have an inconsistent number of bins: ",
        c->Value(bins_s1), " and ", c->Value(bins_p));
  }
  std::vector<tensorflow::shape_inference::DimensionHandle> dims;

  // Evaluate output shape, broadcast all but the last dims.
  // Last dimension represents distribution and can not be broadcasted.
  for (int i = 0; i < rank_p - 1; ++i) {
    auto dim = c->Dim(shape_p, i);
    if (eq_rank_s1_and_p) {
      TF_RETURN_IF_ERROR(
          MergeWithBroadcastStatic(c, dim, c->Dim(shape_s1, i), &dim));
    }
    if (eq_rank_s2_and_p) {
      TF_RETURN_IF_ERROR(
          MergeWithBroadcastStatic(c, dim, c->Dim(shape_s2, i), &dim));
    }
    dims.push_back(dim);
  }
  dims.push_back(c->Dim(shape_s2, rank_s2 - 1));
  c->set_output(0, c->MakeShape(dims));
  return tensorflow::Status::OK();
}

}  // namespace

// A helper class which allows to select a particular row from a 2D tensor
// and enables to apply arbitrary permutations on columns for read-write
// access. This allows for an easy processing of non-monotonic support tensors
// without over-complicating operation specific, be it L2 projection or
// cumulative distribution evaluation.
template <typename T>
class PermutedRow {
 public:
  PermutedRow(typename tensorflow::TTypes<T, 2>::Tensor* tensor_2d,
              int64_t row, std::vector<int64_t>* permutation, int64_t size) :
              tensor_2d_(tensor_2d), row_(row), permutation_(permutation),
              size_(size) {}
  inline T& operator()(int64_t column) {
    if (permutation_->empty()) {
      return (*tensor_2d_)(row_, column);
    }
    return (*tensor_2d_)(row_, (*permutation_)[column]);
  }
  inline T& operator()(int64_t column, bool backwards) {
    if (backwards) {
      column = size_ - column - 1;
    }
    return (*this)(column);
  }
  inline int64_t size() {
    return size_;
  }

 private:
  typename tensorflow::TTypes<T, 2>::Tensor* tensor_2d_;
  int64_t row_;
  std::vector<int64_t>* permutation_;
  int64_t size_;
};

// Updates the current stride and index which are used to evaluate
// a read position for input tensors in the presence of broadcasting.
// Args:
//   dim_size: size of a given dimension of the input tensor;
//   dim_index: current index along a given dimension in the result tensor;
//   stride: the current stride in the input tensor; by definition this is a
//       product of all the higher dimensions than the current one;
//   flat_index: current row in the flat 2D input tensor.
// The function updates stride and flat_index.
void UpdateStrideAndIndex(int64_t dim_size,
                          int64_t dim_index,
                          int64_t* stride,
                          int64_t* flat_index) {
  if (dim_size > 1) {
    *flat_index += (*stride * dim_index);
    *stride *= dim_size;
  }
  // Broadcasting happens when dim_size==1.
  // It requires no updates to stride and flat_index.
}

// Produces a permutation which sorts a given support vector.
// The function is O(n) when the support vector is monotonically
// increasing and is O(n*ln(n)) otherwise.
void Argsort(
    std::vector<int64_t>* permutation,
    const tensorflow::TTypes<const float, 2>::Tensor & support,
    int64_t row, int64_t bins) {
  // At first we check if the support is already monotonically increasing.
  // This helps to speed up the algorithm to O(n) in those cases.
  bool monotonically_increasing = true;
  float last = support(row, 0);
  for (int i = 1; (i < bins) && monotonically_increasing; i++) {
    float next = support(row, i);
    monotonically_increasing = (next >= last);
    last = next;
  }
  if (monotonically_increasing) {
    // Empty permutation is interpreted as a no-op. Shrinking the vector
    // does not cause a heap reallocation. Such zero-size identity
    // representation allows to make sure that not a single heap allocation
    // happens when all the input supports are nicely sorted.
    permutation->resize(0);
  } else {
    if (permutation->empty()) {
      // This is where the first (and the only) heap allocation happens
      // for each worker thread once per op execution time.
      permutation->resize(bins);
      for (int i = 0; i < bins; i++) {
        (*permutation)[i] = i;
      }
    }
    // Finally we produce the permutation.
    auto comparator = [&support, row](int i, int j) {
      float a = support(row, i);
      float b = support(row, j);
      if (a == b) {
        // This makes the sort stable. We need this in order to make the results
        // independent of implementation details of sorting algorithm.
        return i < j;
      }
      return a < b;
    };
    // This line runs an O(n ln n) introsort algorithm.
    std::sort(permutation->begin(), permutation->end(), comparator);
  }
}

// Implements an L2-projection operation onto a target support.
inline void RowL2Project(
    PermutedRow<const float>* support_1d,
    PermutedRow<const float>* weights_1d,
    PermutedRow<const float>* new_support_1d,
    PermutedRow<float>* output_1d) {
  // At this stage all supports are monotonically increasing.
  int64_t bins1 = support_1d->size();
  int64_t bins2 = new_support_1d->size();
  int64_t target_bin = 0;
  float first_ge_value = (*new_support_1d)(0);
  // Zero the output tensor.
  for (int64_t bin = 0; bin < bins2; bin++) {
    (*output_1d)(bin) = 0;
  }
  // Finally, for a given row we go through all input bins and
  // monotonically find respective output bins.
  for (int64_t bin = 0; bin < bins1; bin++) {
    float weight = (*weights_1d)(bin);
    if (weight == 0) {
      continue;
    }
    float bin_value = (*support_1d)(bin);
    while ((target_bin < bins2 - 1) && (first_ge_value < bin_value)) {
      target_bin++;
      first_ge_value = (*new_support_1d)(target_bin);
    }
    if ((target_bin == 0) || (first_ge_value <= bin_value)) {
      // Assign all weight to the last bin if it either matches
      // the value of the previous bin exactly or is the first
      // or the last bin.
      (*output_1d)(target_bin) += weight;
    } else {
      float value1 = (*new_support_1d)(target_bin - 1);
      float value2 = (*new_support_1d)(target_bin);
      float k1 = value2 - bin_value;
      float k2 = bin_value - value1;
      CHECK_GE(k1, 0);
      CHECK_GE(k2, 0);
      float sum_k = k1 + k2;
      if (sum_k > 0) {
        k1 /= sum_k;
        k2 /= sum_k;
      } else {
        k1 = 0.5;
        k2 = 0.5;
      }
      (*output_1d)(target_bin - 1) += weight * k1;
      (*output_1d)(target_bin) += weight * k2;
    }
  }
}

// Evaluates a comulative distribution of the given one on a new support.
inline void RowHardCumulativeProject(
    PermutedRow<const float>* support_1d,
    PermutedRow<const float>* weights_1d,
    PermutedRow<const float>* new_support_1d,
    PermutedRow<float>* output_1d,
    bool reverse) {
  // At this stage all supports are monotonically increasing.
  int64_t bins1 = support_1d->size();
  int64_t bins2 = new_support_1d->size();
  float sumw_less = 0.0;
  int64_t bin_old = 0;
  int64_t bin_new = 0;
  auto less = [reverse] (float a, float b, bool reverese) {
    return reverse ? b < a : a < b;
  };
  auto less_or_equal = [reverse] (float a, float b, bool reverese) {
    return reverse ? b <= a : a <= b;
  };
  while (true) {
    float new_value = (*new_support_1d)(bin_new, reverse);
    while ((bin_old < bins1) &&
        less((*support_1d)(bin_old, reverse), new_value, reverse)) {
      sumw_less += (*weights_1d)(bin_old, reverse);
      bin_old++;
    }
    if (bin_old >= bins1) {
      break;
    }
    float old_value = (*support_1d)(bin_old, reverse);
    while ((bin_new < bins2) &&
        less_or_equal((*new_support_1d)(bin_new, reverse),
            old_value, reverse)) {
      (*output_1d)(bin_new, reverse) = sumw_less;
      bin_new++;
    }
    if (bin_new >= bins2) {
      break;
    }
  }
  for (; bin_new < bins2; bin_new++) {
    (*output_1d)(bin_new, reverse) = sumw_less;
  }
}


REGISTER_OP("ProjectDistribution")
    .Input("support: float32")
    .Input("weights: float32")
    .Input("new_support: float32")
    .Input("method: int32")
    .Output("new_weights: float32")
    .SetShapeFn(ProjectDistributionShapeFn)
    .Doc(R"doc(
Projects one categorical distribution onto another.
)doc");

// Computes the actual projection at session.run time. Runtime complexity
// is O(b) in broadcasted input size if supports are monotonically increasing
// and is O(b*ln(b)) along the bin dimension if supports are not increasing.
// Inputs tensors are: support_old, weights, support_new.
// 1. Input supports can be arbitrary, but the op will run much faster
//    if they are monotonically increasing.
// 2. Ranks of all supports have to be 1 or equal to the rank of weights.
// 3. Broadcasting happens along all dimensions except the last one. This means
//    that sizes of any given dimension (except the last) must be equal or
//    be of size 1.
// 4. support_old and weights have to have an equal last dimension which
//    represents the number of bins.
// 5. support_new can have the last dimension of any size. Output tensor
//    will have a dimension of that size.
// Depending on the supplied scalar `method` tensor the op will execute one
// of the following algorithms:
// 1. L2_project
// 2. Cumulative: result[i] = sum_j weights[j] where support[j] < support[i].
// 3. Cumulative: result[i] = sum_j weights[j] where support[j] > support[i].
void ProjectDistribution::Compute(tensorflow::OpKernelContext* context) {
  // Grab the input tensors.
  auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());

  const tensorflow::Tensor& support_tensor = context->input(0);
  const tensorflow::Tensor& weights_tensor = context->input(1);
  const tensorflow::Tensor& new_support_tensor = context->input(2);
  const tensorflow::Tensor& method_tensor = context->input(3);

  tensorflow::TensorShape new_support_shape = new_support_tensor.shape();
  tensorflow::TensorShape shape_s1 = support_tensor.shape();
  tensorflow::TensorShape shape_s2 = new_support_tensor.shape();
  tensorflow::TensorShape shape_p = weights_tensor.shape();
  tensorflow::TensorShape shape_m = method_tensor.shape();
  int rank_s1 = shape_s1.dims();
  int rank_s2 = shape_s2.dims();
  int rank_p = shape_p.dims();
  int rank_m = shape_m.dims();
  bool eq_rank_s1_and_p = (rank_s1 == rank_p);
  bool eq_rank_s2_and_p = (rank_s2 == rank_p);
  OP_REQUIRES(context, eq_rank_s1_and_p || rank_s1 == 1,
      tensorflow::errors::InvalidArgument(
          "Rank of support has to be 1 or match the rank of weights"));
  OP_REQUIRES(context, eq_rank_s2_and_p || rank_s2 == 1,
      tensorflow::errors::InvalidArgument(
          "Rank of new_support has to be 1 or match the rank of weights"));
  OP_REQUIRES(context, rank_m == 0,
      tensorflow::errors::InvalidArgument(
          "Rank of method must be 0"));
  tensorflow::TensorShape output_shape;
  for (int r = 0; r < rank_p - 1; r++)   {
    int64_t d0 = shape_p.dim_size(r);
    if (eq_rank_s1_and_p) {
      OP_REQUIRES_OK(context,
          MergeWithBroadcastDynamic(d0, shape_s1.dim_size(r), &d0));
    }
    if (eq_rank_s2_and_p) {
      OP_REQUIRES_OK(context,
          MergeWithBroadcastDynamic(d0, shape_s2.dim_size(r), &d0));
    }
    output_shape.AddDim(d0);
  }
  output_shape.AddDim(shape_s2.dim_size(rank_s2 - 1));

  tensorflow::Tensor* output_tensor = nullptr;
  OP_REQUIRES_OK(context,
      context->allocate_output(0, output_shape, &output_tensor));

  // Flatten tensors.
  tensorflow::TTypes<const float, 2>::Tensor support_2d =
      support_tensor.flat_inner_dims<float>();
  tensorflow::TTypes<const float, 2>::Tensor weights_2d =
      weights_tensor.flat_inner_dims<float>();
  tensorflow::TTypes<const float, 2>::Tensor new_support_2d =
      new_support_tensor.flat_inner_dims<float>();
  // tensorflow::TTypes<const int>::Scalar
  int method_id = method_tensor.scalar<int>()();
  tensorflow::TTypes<float, 2>::Tensor output_2d =
      output_tensor->flat_inner_dims<float>();

  OP_REQUIRES(context, (method_id >= 1) && (method_id <= 3),
      tensorflow::errors::InvalidArgument(
          "Method ID must be between [1, 3]"));

  // We define a callback in order to parallelize the op.
  auto DoWork = [method_id, rank_p, &shape_s1, &shape_s2, &shape_p,
                 &output_shape, &support_2d, &weights_2d, &new_support_2d,
                 &output_2d](int64_t start_row, int64_t limit_row) {
    int64_t bins1 = shape_s1.dim_size(shape_s1.dims() - 1);
    int64_t bins2 = shape_s2.dim_size(shape_s2.dims() - 1);
    // Define permutation vectors for the cases when bins are not
    // monotonically increasing. In this case the algorithm's complexity
    // will increase from O(n) to O(n*ln(n)) due to sorting.
    // Note that empty vectors do not have any heap presence, so allocation
    // cost will only be served when at least one non-monotonic support is
    // found.
    std::vector<int64_t> permutation_1;
    std::vector<int64_t> permutation_2;
    for (int64_t row = start_row; row < limit_row; row++) {
      // Convert flat output row indixes into flat source indices by taking
      // into the account broadcasting. This is done by expanding flat output
      // rows into multi-dimensional indices, and computing respective flat
      // row indices for input tensors. Any dimension of size 1 is broadcasted
      // with the exception of the last dimension representing bins.
      int64_t index = row;
      int64_t stride_p = 1;
      int64_t stride_s1 = 1;
      int64_t stride_s2 = 1;
      int64_t index_p = 0;
      int64_t index_s1 = 0;
      int64_t index_s2 = 0;

      // Ignore the last dimension as it contains bins and can not be
      // broadcasted over.
      for (int d = rank_p - 2; d >= 0; d--) {
        int64_t current_dim_size = output_shape.dim_size(d);
        int64_t current_dim_index = index % current_dim_size;
        index /= current_dim_size;
        UpdateStrideAndIndex(shape_p.dim_size(d), current_dim_index,
            &stride_p, &index_p);
        if (shape_s1.dims() > 1) {
          UpdateStrideAndIndex(shape_s1.dim_size(d), current_dim_index,
             &stride_s1, &index_s1);
        }
        if (shape_s2.dims() > 1) {
          UpdateStrideAndIndex(shape_s2.dim_size(d), current_dim_index,
             &stride_s2, &index_s2);
        }
      }
      // In some cases the provided support vectors may not be monotonic, so
      // we need to produce permutation vectors defining positions of support
      // bins in an ascending order.
      Argsort(&permutation_1, support_2d, index_s1, bins1);
      Argsort(&permutation_2, new_support_2d, index_s2, bins2);
      PermutedRow<const float>
          support_v(&support_2d, index_s1, &permutation_1, bins1);
      PermutedRow<const float>
          weights_v(&weights_2d, index_p, &permutation_1, bins1);
      PermutedRow<const float>
          new_support_v(&new_support_2d, index_s2, &permutation_2, bins2);
      PermutedRow<float>
          output_v(&output_2d, row, &permutation_2, bins2);
      // Finally, process a given row depending on the method chosen.
      if (method_id == 1) {
        // Project one distribution onto another.
        RowL2Project(
            &support_v, &weights_v, &new_support_v, &output_v);
      } else if ((method_id == 2) || (method_id == 3)) {
        // Evaluate a cumulative of (support_v, weights_v) on new_support_v.
        bool reverse = (method_id == 3);
        RowHardCumulativeProject(
            &support_v, &weights_v, &new_support_v, &output_v, reverse);
      }
    }
  };
  // A rough estimate of CPU clocks per row assuming monotonically
  // increasing supports. Coefficients were obtained by
  // profiling the code on Intel(R) Xeon(R) CPU E5-1650 v4 @ 3.60GHz.
  // Note that it is better to underestimate the values slightly than to
  // overestimate as the later could lead to an excessive thread congestion.
  // For this reason the estimate is produced for the simplest linear case.
  const int64_t sum_bins = (support_2d.dimension(1) +
                            new_support_2d.dimension(1));
  const int64_t cost = 40 * sum_bins + 16 * rank_p;
  tensorflow::Shard(worker_threads.num_threads, worker_threads.workers,
      output_2d.dimension(0), cost, DoWork);
}

REGISTER_KERNEL_BUILDER(
    Name("ProjectDistribution")
        .Device(tensorflow::DEVICE_CPU),
    ProjectDistribution);

}  // namespace rl
}  // namespace deepmind
