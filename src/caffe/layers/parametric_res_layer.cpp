#include <cfloat>
#include <vector>

#include "caffe/layers/parametric_res_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ParametricResLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
    this->blobs_[0].mutable_cpu_data()[0] = 0; // init to zero -- need to think how to pass parameter here...
  }
}

template <typename Dtype>
void ParametricResLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK(bottom[1]->shape() == bottom[0]->shape()) << "inputs must have same shape";
  top[0]->ReshapeLike(*bottom[0]);
  // reshape internals
  m_.ReshapeLike(*bottom[0]);
  tx1_.ReshapeLike(*bottom[0]);
  tx2_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ParametricResLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype theta = this->blobs_[0].cpu_data()[0]; // get the parameter
  const Dtype* x1 = bottom[0]->cpu_data();
  const Dtype* x2 = bottom[1]->cpu_data();
  Dtype* tx1 = tx1_.mutable_cpu_data();
  Dtype* tx2 = tx2_.mutable_cpu_data();
  caffe_copy(count, x1, tx1);
  caffe_scal(count, theta, tx1); // tx1 <- \theta x1
  caffe_copy(count, x2, tx2);
  caffe_scal(count, theta, tx2); // tx2 <- \theta x2
  // find max
  Dtype* m = m_.mutable_cpu_data();
  for (int i = 0; i < count; i++) {
    m[i] = max(tx1[i], tx2[i]);
  }
  caffe_sub(count, tx1, m, tx1);  // tx1 <- \theta x1 - m
  caffe_sub(count, tx2, m, tx2);  // tx2 <- \theta x2 - m
  // store the difference in m. for gradient
  caffe_sub(count, tx1, tx2, m);  // m <- \theta x1 - \theta x2
  // exp
  caffe_exp(count, tx1, tx1);
  caffe_exp(count, tx2, tx2);
  // denominator
  Dtype* denom = denom_.mutable_cpu_data();
  caffe_add(count, tx1, tx2, denom);
  Dtype* y = top[0]->mutable_cpu_data();
  for (int i = 0; i < count; i++) {
    y[i] = (x1[i]*tx1[i] + x2[i]*tx2[i]) / denom[i];
  }
}

template <typename Dtype>
void ParametricResLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(ParametricResLayer);
#endif

INSTANTIATE_CLASS(ParametricResLayer);
REGISTER_LAYER_CLASS(ParametricRes);

}; // namespace caffe

