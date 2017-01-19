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
    this->blobs_[0].mutable_cpu_data()[0] = 0; // init to zero
  }
}

template <typename Dtype>
void ParametricResLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK(bottom[1]->shape() == bottom[0]->shape()) << "inpts must have same shape";
  top[0]->ReshapeLike(*bottom[0]);
  // reshape internals

}

template <typename Dtype>
void ParametricResLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
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

