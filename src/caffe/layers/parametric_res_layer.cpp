#include <cfloat>
#include <vector>

#include "caffe/filler.hpp"

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
    // read the filler from PReLU params.
    PReLUParameter prelu_param = this->layer_param().prelu_param();
    shared_ptr<Filler<Dtype> > filler;
    if (prelu_param.has_filler()) {
      filler.reset(GetFiller<Dtype>(prelu_param.filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(1.0);
      filler.reset(GetFiller<Dtype>(filler_param));
    }
    filler->Fill(this->blobs_[0].get());
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void ParametricResLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK(bottom[1]->shape() == bottom[0]->shape()) << "inputs must have same shape";
  top[0]->ReshapeLike(*bottom[0]);
  // reshape internals
  m_.ReshapeLike(*bottom[0]);
  buff_.ReshapeLike(*bottom[0]);
  denom_.ReshapeLike(*bottom[0]);
  tx1_.ReshapeLike(*bottom[0]);
  tx2_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ParametricResLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype theta = this->blobs_[0]->cpu_data()[0]; // get the parameter
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
    m[i] = std::max(tx1[i], tx2[i]);
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
  Dtype* buff = buff_.mutable_cpu_data();
  caffe_mul(count, x1, tx1, buff);
  caffe_mul(count, x2, tx2, y);
  caffe_add(count, buff, y, y);
  caffe_div(count, y, denom, y);
}

template <typename Dtype>
void ParametricResLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // after forward pass,
  const int count = bottom[0]->count();
  const Dtype* x1 = bottom[0]->cpu_data();
  const Dtype* x2 = bottom[1]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* tx1 = tx1_.mutable_cpu_data();
  Dtype* tx2 = tx2_.mutable_cpu_data();
  Dtype* denom = denom_.mutable_cpu_data();
  Dtype* buff = buff_.mutable_cpu_data();
  Dtype* m = m_.mutable_cpu_data(); // after forward pass m <- \theta x1 - \theta x2
  // need denominator squared
  caffe_sqr(count, denom, denom);
  caffe_mul(count, tx1, tx2, buff); // buff <- exp( \theta x1 + \theta x2 - 2m )
  if (propagate_down[0]) {
    // gradient w.r.t x1
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_add_scalar(count, Dtype(1), m);
    caffe_mul(count, m, buff, bottom_diff);
    caffe_sqr(count, tx1, tx1);
    caffe_add(count, bottom_diff, tx1, bottom_diff);
    caffe_div(count, bottom_diff, denom, bottom_diff);
    // finally take into account the top diff
    caffe_mul(count, bottom_diff, top_diff, bottom_diff);
  }
  if (propagate_down[1]) {
    // gradient w.r.t x1
    Dtype* bottom_diff = bottom[1]->mutable_cpu_diff();
    caffe_scal(count, Dtype(-1), m);
    caffe_add_scalar(count, Dtype(2), m); // from tx1-tx2+1 --> tx2-tx1+1
    caffe_mul(count, m, buff, bottom_diff);
    caffe_sqr(count, tx2, tx2);
    caffe_add(count, bottom_diff, tx2, bottom_diff);
    caffe_div(count, bottom_diff, denom, bottom_diff);
    // finally take into account the top diff
    caffe_mul(count, bottom_diff, top_diff, bottom_diff);
  }
  // gradient w.r.t theta
  caffe_div(count, buff, denom, buff);  // buff <- exp( \theta x1 + \theta x2 - 2m ) / ()^2
  caffe_sub(count, x1, x2, m);
  caffe_sqr(count, m, m);  // m <- (x1-x2)^2
  caffe_mul(count, m, top_diff, m); // take into account the top diff
  Dtype* theta_diff = this->blobs_[0]->mutable_cpu_diff();
  theta_diff[0] = caffe_cpu_dot(count, m, buff);
}

#ifdef CPU_ONLY
STUB_GPU(ParametricResLayer);
#endif

INSTANTIATE_CLASS(ParametricResLayer);
REGISTER_LAYER_CLASS(ParametricRes);

}; // namespace caffe

