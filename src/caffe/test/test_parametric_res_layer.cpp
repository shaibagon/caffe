#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/parametric_res_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ParametricResLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ParametricResLayerTest()
      : blob_bottom_x1_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_x2_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_min(-1);
    filler_param.set_max(1);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_x1_);
    filler.Fill(this->blob_bottom_x2_);
    blob_bottom_vec_.push_back(blob_bottom_x1_);
    blob_bottom_vec_.push_back(blob_bottom_x2_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ParametricResLayerTest() {
    delete blob_bottom_x1_;
    delete blob_bottom_x2_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_x1_;
  Blob<Dtype>* const blob_bottom_x2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ParametricResLayerTest, TestDtypesAndDevices);

TYPED_TEST(ParametricResLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<ParametricResLayer<Dtype> > layer(
      new ParametricResLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(ParametricResLayerTest, TestMax) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler;
  filler.set_type("constant");
  // for max test, set the internal param to very high value
  filler.set_value(100);
  LayerParameter layer_param;
  layer_param.mutable_prelu_param()->set_filler(filler);
  shared_ptr<ParametricResLayer<Dtype> > layer(
      new ParametricResLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* x1 = this->blob_bottom_x1_->cpu_data();
  const Dtype* x2 = this->blob_bottom_x2_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], std::max(x1[i], x2[i]), 1e-4);
  }
}

TYPED_TEST(ParametricResLayerTest, TestMin) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler;
  filler.set_type("constant");
  // for min test, set the internal param to very low value
  filler.set_value(-100);
  LayerParameter layer_param;
  layer_param.mutable_prelu_param()->set_filler(filler);
  shared_ptr<ParametricResLayer<Dtype> > layer(
      new ParametricResLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* x1 = this->blob_bottom_x1_->cpu_data();
  const Dtype* x2 = this->blob_bottom_x2_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], std::min(x1[i], x2[i]), 1e-4);
  }
}

TYPED_TEST(ParametricResLayerTest, TestMean) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler;
  filler.set_type("constant");
  // for mean test, set the internal param to zero
  filler.set_value(0);
  LayerParameter layer_param;
  layer_param.mutable_prelu_param()->set_filler(filler);
  shared_ptr<ParametricResLayer<Dtype> > layer(
      new ParametricResLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* x1 = this->blob_bottom_x1_->cpu_data();
  const Dtype* x2 = this->blob_bottom_x2_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], 0.5*(x1[i] + x2[i]), 1e-4);
  }
}

TYPED_TEST(ParametricResLayerTest, TestTheta1) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler;
  filler.set_type("constant");
  // test intermediate value of theta
  filler.set_value(1);
  LayerParameter layer_param;
  layer_param.mutable_prelu_param()->set_filler(filler);
  shared_ptr<ParametricResLayer<Dtype> > layer(
      new ParametricResLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* x1 = this->blob_bottom_x1_->cpu_data();
  const Dtype* x2 = this->blob_bottom_x2_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], (x1[i]*std::exp(x1[i]) + x2[i]*std::exp(x2[i]))/(std::exp(x1[i]) + std::exp(x2[i])), 1e-4);
  }
}

TYPED_TEST(ParametricResLayerTest, TestMaxGradient) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler;
  filler.set_type("constant");
  filler.set_value(10);
  LayerParameter layer_param;
  layer_param.mutable_prelu_param()->set_filler(filler);
  ParametricResLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 5e-3);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ParametricResLayerTest, TestMinGradient) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler;
  filler.set_type("constant");
  filler.set_value(-10);
  LayerParameter layer_param;
  layer_param.mutable_prelu_param()->set_filler(filler);
  ParametricResLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 5e-3);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ParametricResLayerTest, TestMeanGradient) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler;
  filler.set_type("constant");
  filler.set_value(0);
  LayerParameter layer_param;
  layer_param.mutable_prelu_param()->set_filler(filler);
  ParametricResLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 5e-3);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ParametricResLayerTest, TestTheta1Gradient) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler;
  filler.set_type("constant");
  filler.set_value(1);
  LayerParameter layer_param;
  layer_param.mutable_prelu_param()->set_filler(filler);
  ParametricResLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 5e-3);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
