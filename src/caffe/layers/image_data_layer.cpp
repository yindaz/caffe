#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

extern DataProcess ExternData;

namespace caffe {

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label;
  while (infile >> filename >> label) {
    lines_.push_back(std::make_pair(filename, label));
  }	

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";
	
	// initial positive and negative
	ExternData.PosIDs.clear();
	ExternData.NegIDs.clear();
	for ( int i = 0; i<lines_.size(); i++)
	{
		if ( lines_[i].second<-0.5) ExternData.NegIDs.push_back(i);
		else	ExternData.PosIDs.push_back(i);
	}
	ExternData.pid = 0;
	ExternData.nid = 0;
	LOG(INFO) << "Pos: " << ExternData.PosIDs.size();
	LOG(INFO) << "Neg: " << ExternData.NegIDs.size();

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  CHECK(ReadImageToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
                         new_height, new_width, &datum));
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size,
                                 crop_size);
  } else {
    (*top)[0]->Reshape(batch_size, datum.channels(), datum.height(),
                       datum.width());
    this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(),
        datum.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  (*top)[1]->Reshape(batch_size, 1, 1, 1);
  this->prefetch_label_.Reshape(batch_size, 1, 1, 1);
  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();

	//// get all keys from dataset
	if (ExternData.InitialID == 0 )
	{
		ExternData.NumberData = lines_.size();

		// ExternData.Weight.assign( ExternData.NumberData, 0);
		ExternData.Label.assign( ExternData.NumberData, 0);
		// Key.assign(NumberData, MDB_val());
		ExternData.IsNegative.assign( ExternData.NumberData, false);
		ExternData.NumberInitialed = true;
		ExternData.InitialID = ExternData.NumberData;
		LOG(INFO) << "Data initialized: " << ExternData.NumberData;
	}
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageDataLayer<Dtype>::InternalThreadEntry() {
	int NumberData = lines_.size();
  bool DATAUPDATE = true;
  if (NumberData != ExternData.NumberData)
  {
	  //LOG(INFO) << "Loading testing data";
	  DATAUPDATE = false;
  }
  else{
	  //LOG(INFO) << "Loading training data";
  }

  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();

	//ExternData.ResetDataLocal(batch_size);
	vector<int> IDs;
	vector<int> temp;
	int local_lines_id;
  if (DATAUPDATE)
  {
	  ExternData.SelectedDataID.assign(batch_size, -1);
		ExternData.ReturnDataIDs( batch_size, ExternData.pnratio, IDs);
		//LOG(INFO) << "Size of IDs: " << IDs.size();
		for ( int item_id = 0; item_id < batch_size; ++item_id)
		{
			//LOG(INFO) << item_id << " " << IDs[item_id];			
			if (!ReadImageToDatum(lines_[IDs[item_id]].first,
          lines_[IDs[item_id]].second,
          new_height, new_width, &datum)) 
			{
				ExternData.ReturnDataIDs(1, 1, temp);
				//LOG(INFO) << "Corrupted data: " << temp.size() << " " << temp[0];				
				while (!ReadImageToDatum(lines_[temp[0]].first,
          lines_[temp[0]].second,
          new_height, new_width, &datum))
				{					
					ExternData.ReturnDataIDs(1, 1, temp);					
				}
				local_lines_id = temp[0];
		  }
			else
			{
				local_lines_id = IDs[item_id];
			}
			//LOG(INFO) << "lines_id: " << local_lines_id;
			this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);
    	top_label[item_id] = datum.label();
			int curID = local_lines_id;
			//LOG(INFO) << item_id << " " << curID << " " << top_label[item_id] << " " << ExternData.ACTIVATE;
			ExternData.UpdateGlobalLabel( curID, top_label[item_id]);
			ExternData.UpdateSelectedID( item_id, curID);

		}  

	}

	
	/*
  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    CHECK_GT(lines_size, lines_id_);
    if (!ReadImageToDatum(lines_[lines_id_].first,
          lines_[lines_id_].second,
          new_height, new_width, &datum)) {
			lines_id_++;
			item_id--;
      continue;
    }

		//if (datum.label() < -0.5)
		//{
		//	lines_id_++;
		//	item_id--;
		//	continue;
		//}
    // Apply transformations (mirror, crop...) to the data
    this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);

    top_label[item_id] = datum.label();

    if (DATAUPDATE)
    {
			int curID = lines_id_;
			LOG(INFO) << item_id << " " << curID << " " << top_label[item_id] << " " << ExternData.ACTIVATE;
			ExternData.UpdateGlobalLabel( curID, top_label[item_id]);
			ExternData.UpdateSelectedID( item_id, curID);
    }

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        //ShuffleImages();
      }
    }
  }
	*/	
	
	if (DATAUPDATE)
  {
//	  LOG(INFO) << ExternData.ReadyToRead;
//	  while(!ExternData.ReadyToRead && ExternData.CurrentID==255)
//	  {
//		  ExternData.ReadyToRead = true;
//		  usleep(1000);
//		  LOG(INFO) << "*";
//	  }
//	  ExternData.ReadyToRead = true;
//	  LOG(INFO) << ExternData.ReadyToRead;
//	  LOG(INFO) << ExternData.readReadyToRead();
	  ExternData.setReadyToRead(true);
//	  LOG(INFO) << ExternData.readReadyToRead();
	  ExternData.IterCounter ++;
  }
}

INSTANTIATE_CLASS(ImageDataLayer);

}  // namespace caffe
