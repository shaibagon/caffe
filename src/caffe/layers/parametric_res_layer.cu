#include <cfloat>
#include <vector>

#include "caffe/layers/parametric_res_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void EltwiseMax(const int nthreads, const Dtype* tx1, const Dtype* tx2, Dtype* m) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    m[index] = (tx1[index]>tx2[index]) ? tx1[index] : tx2[index];
  }
}

template <typename Dtype>
void ParametricResLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype theta = this->blobs_[0]->cpu_data()[0]; // get the parameter
  const Dtype* x1 = bottom[0]->gpu_data();
  const Dtype* x2 = bottom[1]->gpu_data();
  Dtype* tx1 = tx1_.mutable_gpu_data();
  Dtype* tx2 = tx2_.mutable_gpu_data();
  caffe_gpu_memcpy(count*sizeof(Dtype), x1, tx1);
  caffe_gpu_scal(count, theta, tx1); // tx1 <- \theta x1
  caffe_gpu_memcpy(count*sizeof(Dtype), x2, tx2);
  caffe_gpu_scal(count, theta, tx2); // tx2 <- \theta x2
  // find max
  Dtype* m = m_.mutable_gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  EltwiseMax<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, tx1, tx2, m);
  caffe_gpu_sub(count, tx1, m, tx1);  // tx1 <- \theta x1 - m
  caffe_gpu_sub(count, tx2, m, tx2);  // tx2 <- \theta x2 - m
  // store the difference in m. for gradient
  caffe_gpu_sub(count, tx1, tx2, m);  // m <- \theta x1 - \theta x2
  // exp
  caffe_gpu_exp(count, tx1, tx1);
  caffe_gpu_exp(count, tx2, tx2);
  // denominator
  Dtype* denom = denom_.mutable_gpu_data();
  caffe_gpu_add(count, tx1, tx2, denom);
  Dtype* y = top[0]->mutable_gpu_data();
  Dtype* buff = buff_.mutable_gpu_data();
  caffe_gpu_mul(count, x1, tx1, buff);
  caffe_gpu_mul(count, x2, tx2, y);
  caffe_gpu_add(count, buff, y, y);
  caffe_gpu_div(count, y, denom, y);
}

template <typename Dtype>
void ParametricResLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // after forward pass,
  const int count = bottom[0]->count();
  const Dtype* x1 = bottom[0]->gpu_data();
  const Dtype* x2 = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* tx1 = tx1_.mutable_gpu_data();
  Dtype* tx2 = tx2_.mutable_gpu_data();
  Dtype* denom = denom_.mutable_gpu_data();
  Dtype* buff = buff_.mutable_gpu_data();
  Dtype* m = m_.mutable_gpu_data(); // after forward pass m <- \theta x1 - \theta x2
  // need denominator squared
  caffe_gpu_mul(count, denom, denom, denom);
  caffe_gpu_mul(count, tx1, tx2, buff); // buff <- exp( \theta x1 + \theta x2 - 2m )
  if (propagate_down[0]) {
    // gradient w.r.t x1
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_add_scalar(count, Dtype(1), m);
    caffe_gpu_mul(count, m, buff, bottom_diff);
    caffe_gpu_mul(count, tx1, tx1, tx1);
    caffe_gpu_add(count, bottom_diff, tx1, bottom_diff);
    caffe_gpu_div(count, bottom_diff, denom, bottom_diff);
    // finally take into account the top diff
    caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
  }
  if (propagate_down[1]) {
    // gradient w.r.t x1
    Dtype* bottom_diff = bottom[1]->mutable_gpu_diff();
    caffe_gpu_scal(count, Dtype(-1), m);
    caffe_gpu_add_scalar(count, Dtype(2), m); // from tx1-tx2+1 --> tx2-tx1+1
    caffe_gpu_mul(count, m, buff, bottom_diff);
    caffe_gpu_mul(count, tx2, tx2, tx2);
    caffe_gpu_add(count, bottom_diff, tx2, bottom_diff);
    caffe_gpu_div(count, bottom_diff, denom, bottom_diff);
    // finally take into account the top diff
    caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
  }
  // gradient w.r.t theta
  caffe_gpu_div(count, buff, denom, buff);  // buff <- exp( \theta x1 + \theta x2 - 2m ) / ()^2
  caffe_gpu_sub(count, x1, x2, m);
  caffe_gpu_mul(count, m, m, m);  // m <- (x1-x2)^2
  caffe_gpu_mul(count, m, top_diff, m); // take into account the top diff
  Dtype* theta_diff = this->blobs_[0]->mutable_cpu_diff(); // not gpu_diff here, gpu_dot pushes result to host apparently
  caffe_gpu_dot(count, m, buff, theta_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(ParametricResLayer);

} // namespace caffe

