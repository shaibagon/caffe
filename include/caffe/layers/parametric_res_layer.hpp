#ifndef CAFFE_PARAMETRIC_RES_LAYER_HPP_
#define CAFFE_PARAMETRIC_RES_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

/**
 * @brief Parameterized residual branch merging @f$
 *        y_i = (x1_i exp(\theta x1_i) + x2_i exp(\theta x2_i))/ \max(0, x_i) + a_i \min(0, x_i)
 *        @f$.
 */
template <typename Dtype>
class ParametricResLayer : public Layer<Dtype> {
 public:
  /**
   */
  explicit ParametricResLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ParametricRes"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times ...) @f$
   *      the input @f$ x1 @f$
   *   -# @f$ (N \times C \times ...) @f$
   *      the input @f$ x2 @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times ...) @f$
   *      the computed outputs for each element @f$i@f$ @f$
   *        y_i = (x1_i exp(\theta x1_i) + x2_i exp(\theta x2_i))/ \max(0, x_i) + a_i \min(0, x_i)
   *      @f$.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
//  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the PReLU inputs.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (N \times C \times ...) @f$
   *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
   *      with respect to computed outputs @f$ y @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times ...) @f$
   *      the inputs @f$ x1 @f$; For each channel @f$i@f$, backward fills their
   *      diff with gradients @f$
   *        \frac{\partial E}{\partial x_i} = \left\{
   *        \begin{array}{lr}
   *            a_i \frac{\partial E}{\partial y_i} & \mathrm{if} \; x_i \le 0 \\
   *            \frac{\partial E}{\partial y_i} & \mathrm{if} \; x_i > 0
   *        \end{array} \right.
   *      @f$.
   *      If param_propagate_down_[0] is true, it fills the diff with gradients
   *      @f$
   *        \frac{\partial E}{\partial a_i} = \left\{
   *        \begin{array}{lr}
   *            \sum_{x_i} x_i \frac{\partial E}{\partial y_i} & \mathrm{if} \; x_i \le 0 \\
   *            0 & \mathrm{if} \; x_i > 0
   *        \end{array} \right.
   *      @f$.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
//  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> m_;      // max of x1_i, x2_i (for numerically stable computation)
  Blob<Dtype> buff_;   // extra buffer
  Blob<Dtype> denom_;  // caching the denominator
  Blob<Dtype> tx1_;    // temporary buffer for
  Blob<Dtype> tx2_;    // memory for in-place computation
};

}  // namespace caffe

#endif  // CAFFE_PARAMETRIC_RES_LAYER_HPP_
