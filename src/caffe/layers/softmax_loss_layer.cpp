#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/data_process.hpp"

extern DataProcess ExternData;

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, &softmax_top_vec_);
  if (top->size() >= 2) {
    // softmax output
    (*top)[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {

	//LOG(INFO) << "SoftmaxWithLossLayer Forward";

  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();



  int num = prob_.num();
  int dim = prob_.count() / num;
  int spatial_dim = prob_.height() * prob_.width();


//	if (ExternData.TESTING)
//	{
//		LOG(INFO) << "Testing Softmax";
//	}
//	else
//	{
//		LOG(INFO) << "Training Softmax";
//		for (int i = 0; i<num; i++)
//		{
//			LOG(INFO) << label[i] << " " <<ExternData.MiniBatchLabel[i];
//		}
//	}


  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
    	if (!ExternData.TESTING)
    	{
		// check

			int rec = ExternData.MiniBatchLabel[i];
			int ori = static_cast<int>(label[i * spatial_dim + j]);
//			if ( rec != ori && rec != -ori-1)
//			{
//				LOG(INFO) << "Warning! Data ERROR: ";
//				LOG(INFO) << ExternData.MiniBatchDataID[i] <<" "<<rec<<" "<<ori<<" "<<ExternData.CurrentID;
//			}
			int l = ExternData.MiniBatchLabel[i];
			//      int l = static_cast<int>(label[i * spatial_dim + j]);
			//      Dtype r = label[i * spatial_dim + j] - l;

			if (ExternData.MiniBatchIsneg[i]) // negative label 0.1
			{
			loss -= log(std::max( 1 - prob_data[i * dim + l * spatial_dim + j],
							   Dtype(FLT_MIN)));
			}
			else
			{
			loss -= log(std::max(prob_data[i * dim + l * spatial_dim + j],
							   Dtype(FLT_MIN)));
			}
    	}
    	else
    	{
    		      loss -= log(std::max(prob_data[i * dim +
    		          static_cast<int>(label[i * spatial_dim + j]) * spatial_dim + j],
    		                           Dtype(FLT_MIN)));
    	}


    }
  }
  (*top)[0]->mutable_cpu_data()[0] = loss / num / spatial_dim;
  if (top->size() == 2) {
    (*top)[1]->ShareData(prob_);
  }
	//LOG(INFO) << "SoftmaxWithLossLayer Forward end";
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {

	//LOG(INFO) << "SoftmaxWithLossLayer Backward";
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = (*bottom)[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    int spatial_dim = prob_.height() * prob_.width();
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
    	  int l = ExternData.MiniBatchLabel[i];
//        int l = static_cast<int>(label[i * spatial_dim + j]);
//        Dtype r = label[i * spatial_dim + j] - l;
        if (ExternData.MiniBatchIsneg[i]) // negative label 0.1
        {
          for (int k = 0; k < dim; ++k)
          {
            if ( k != l )
            {
              bottom_diff[i * dim + k * spatial_dim + j] = 0;
            }
          }
        }
        else
        {
          bottom_diff[i * dim + l * spatial_dim + j] -= 1;
        }

      }
    }
    // Scale gradient
//    const Dtype loss_weight = top[0]->cpu_diff()[0];
//    caffe_scal(prob_.count(), loss_weight / num / spatial_dim, bottom_diff);
    // Yinda: gradient not normalized by num, will be done later after outlier detection
  }
	//LOG(INFO) << "SoftmaxWithLossLayer backward end";
}

//template <typename Dtype>
//void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
//    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
//  // The forward pass computes the softmax prob values.
//  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
//  const Dtype* prob_data = prob_.cpu_data();
//  const Dtype* label = bottom[1]->cpu_data();
//  int num = prob_.num();
//  int dim = prob_.count() / num;
//  int spatial_dim = prob_.height() * prob_.width();
//  Dtype loss = 0;
//  for (int i = 0; i < num; ++i) {
//    for (int j = 0; j < spatial_dim; j++) {
//      loss -= log(std::max(prob_data[i * dim +
//          static_cast<int>(label[i * spatial_dim + j]) * spatial_dim + j],
//                           Dtype(FLT_MIN)));
//    }
//  }
//  (*top)[0]->mutable_cpu_data()[0] = loss / num / spatial_dim;
//  if (top->size() == 2) {
//    (*top)[1]->ShareData(prob_);
//  }
//}
//
//template <typename Dtype>
//void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
//    const vector<bool>& propagate_down,
//    vector<Blob<Dtype>*>* bottom) {
//  if (propagate_down[1]) {
//    LOG(FATAL) << this->type_name()
//               << " Layer cannot backpropagate to label inputs.";
//  }
//  if (propagate_down[0]) {
//    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
//    const Dtype* prob_data = prob_.cpu_data();
//    caffe_copy(prob_.count(), prob_data, bottom_diff);
//    const Dtype* label = (*bottom)[1]->cpu_data();
//    int num = prob_.num();
//    int dim = prob_.count() / num;
//    int spatial_dim = prob_.height() * prob_.width();
//    for (int i = 0; i < num; ++i) {
//      for (int j = 0; j < spatial_dim; ++j) {
//        bottom_diff[i * dim + static_cast<int>(label[i * spatial_dim + j])
//            * spatial_dim + j] -= 1;
//      }
//    }
//    // Scale gradient
//    const Dtype loss_weight = top[0]->cpu_diff()[0];
//    caffe_scal(prob_.count(), loss_weight / num / spatial_dim, bottom_diff);
//  }
//}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);


}  // namespace caffe
