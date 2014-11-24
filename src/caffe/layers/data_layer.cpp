#include <leveldb/db.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

DataProcess ExternData;

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::~DataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    break;  // do nothing
  case DataParameter_DB_LMDB:
    mdb_cursor_close(mdb_cursor_);
    mdb_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_txn_);
    mdb_env_close(mdb_env_);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Initialize DB
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    {
    leveldb::DB* db_temp;
    leveldb::Options options = GetLevelDBOptions();
    options.create_if_missing = false;
    LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
    leveldb::Status status = leveldb::DB::Open(
        options, this->layer_param_.data_param().source(), &db_temp);
    CHECK(status.ok()) << "Failed to open leveldb "
                       << this->layer_param_.data_param().source() << std::endl
                       << status.ToString();
    db_.reset(db_temp);
    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    iter_->SeekToFirst();
    }
    break;
  case DataParameter_DB_LMDB:
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
		LOG(INFO) << "Open: " << this->layer_param_.data_param().source().c_str();
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      switch (this->layer_param_.data_param().backend()) {
      case DataParameter_DB_LEVELDB:
        iter_->Next();
        if (!iter_->Valid()) {
          iter_->SeekToFirst();
        }
        break;
      case DataParameter_DB_LMDB:
        if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
            != MDB_SUCCESS) {
          CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
                   MDB_FIRST), MDB_SUCCESS);
        }
        break;
      default:
        LOG(FATAL) << "Unknown database backend";
      }
    }
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    datum.ParseFromString(iter_->value().ToString());
    break;
  case DataParameter_DB_LMDB:
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // image
  int crop_size = this->layer_param_.transform_param().crop_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(this->layer_param_.data_param().batch_size(),
                       datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), crop_size, crop_size);
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.data_param().batch_size(), datum.channels(),
        datum.height(), datum.width());
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), datum.height(), datum.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  if (this->output_labels_) {
    (*top)[1]->Reshape(this->layer_param_.data_param().batch_size(), 1, 1, 1);
    this->prefetch_label_.Reshape(this->layer_param_.data_param().batch_size(),
        1, 1, 1);
  }
  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();

  // get all keys from dataset
	if (ExternData.InitialID == 0 )
	{
	  ExternData.SetupDataNumber( mdb_env_);
	  ExternData.SetupLMDB( mdb_dbi_, mdb_txn_);
	  mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST);
	  
		ExternData.PosIDs.clear();
		ExternData.NegIDs.clear();
		int label;
	  for (int i = 0; i < 100000/*ExternData.NumberData*/; i++)
	  {
			CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
			ExternData.Key[i] = mdb_key_;
      datum.ParseFromArray(mdb_value_.mv_data,
          mdb_value_.mv_size);	
			label = datum.label();
			if (label < -0.5)
			{
				ExternData.Label[i] = -label-1;
				ExternData.IsNegative[i] = true;
				//ExternData.NegIDs.push_back(i);
			}
			else
			{
				ExternData.Label[i] = label;
				ExternData.IsNegative[i] = false;
				//ExternData.PosIDs.push_back(i);
			}	
			if (label == ExternData.classid)
			{
				ExternData.PosIDs.push_back(i);
			}
			else if ( label == -ExternData.classid-1 || label>0)
			{
				ExternData.NegIDs.push_back(i);
			}	
			mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT);			
	  }
	  mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST);
	  ExternData.InitialID = ExternData.NumberData;
	  LOG(INFO) << "Key initialized: " << ExternData.NumberData;
	  //LOG(INFO) << (char *) ExternData.Key[0].mv_data;
	  //LOG(INFO) << (char *) ExternData.Key[1].mv_data;
	  //LOG(INFO) << (char *) ExternData.Key[ExternData.InitialID-1].mv_data;
	}
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void DataLayer<Dtype>::InternalThreadEntry() {
  // detect training or testing, cannot have the same #samples
  MDB_stat stat;
  mdb_env_stat( mdb_env_, &stat);
  int NumberData = stat.ms_entries;
  bool DATAUPDATE = true;
  if (NumberData != ExternData.NumberData)
  {
	  DATAUPDATE = false;
  }

  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.data_param().batch_size();
	
  if (DATAUPDATE)
  {
	  ExternData.SelectedDataID.clear();
		ExternData.ReturnDataIDs( batch_size, ExternData.pnratio, ExternData.SelectedDataID);	
	}
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
		if (DATAUPDATE)
		{
		  mdb_get	(	mdb_txn_, mdb_dbi_, 
								&ExternData.Key[ExternData.SelectedDataID[item_id]],
								&mdb_value_ );	
			datum.ParseFromArray(mdb_value_.mv_data,
		        mdb_value_.mv_size);
      LOG(INFO) <<  ExternData.Label[ExternData.SelectedDataID[item_id]]<<" "<<datum.label();
		}
		else
		{
			switch (this->layer_param_.data_param().backend()) {
		  case DataParameter_DB_LEVELDB:
		    CHECK(iter_);
		    CHECK(iter_->Valid());
		    datum.ParseFromString(iter_->value().ToString());
		    break;
		  case DataParameter_DB_LMDB:
		    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
		            &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
		    datum.ParseFromArray(mdb_value_.mv_data,
		        mdb_value_.mv_size);
		    break;
		  default:
		    LOG(FATAL) << "Unknown database backend";
		  }
		}

    // Apply data transformations (mirror, scale, crop...)
    this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);

    if (this->output_labels_) {
      top_label[item_id] = datum.label();
			if (top_label[item_id] != ExternData.classid)
			{
				top_label[item_id] = -ExternData.classid-1;
			}
    }

    if (DATAUPDATE)
    {
			ExternData.UpdateSelectedID( item_id, ExternData.SelectedDataID[item_id]);
    }
		else
		{
	    // go to the next iter
		  switch (this->layer_param_.data_param().backend()) {
		  case DataParameter_DB_LEVELDB:
		    iter_->Next();
		    if (!iter_->Valid()) {
		      // We have reached the end. Restart from the first.
		      DLOG(INFO) << "Restarting data prefetching from start.";
		      iter_->SeekToFirst();
		    }
		    break;
		  case DataParameter_DB_LMDB:
		    if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
		            &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
		      // We have reached the end. Restart from the first.
		      DLOG(INFO) << "Restarting data prefetching from start.";
		      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
		              &mdb_value_, MDB_FIRST), MDB_SUCCESS);
		    }
		    break;
		  default:
		    LOG(FATAL) << "Unknown database backend";
		  }
		}
  }

  if (DATAUPDATE)
  {
	  ExternData.setReadyToRead(true);
	  ExternData.IterCounter ++;
  }
}

/*
// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void DataLayer<Dtype>::InternalThreadEntry() {
  // detect training or testing, cannot have the same #samples
  MDB_stat stat;
  mdb_env_stat( mdb_env_, &stat);
  int NumberData = stat.ms_entries;
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
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.data_param().batch_size();

  //ExternData.ResetDataLocal(batch_size);
  if (DATAUPDATE)
  {
	  ExternData.SelectedDataID.assign(batch_size, -1);
  }

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      CHECK(iter_);
      CHECK(iter_->Valid());
      datum.ParseFromString(iter_->value().ToString());
      break;
    case DataParameter_DB_LMDB:
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
      datum.ParseFromArray(mdb_value_.mv_data,
          mdb_value_.mv_size);
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }

    // Apply data transformations (mirror, scale, crop...)
    this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);

    if (this->output_labels_) {
      top_label[item_id] = datum.label();
			if (top_label[item_id]<-0.5 && top_label[item_id]!=-1)
			{
				//LOG(INFO) << "Jump: " << top_label[item_id];
				item_id --;
				// go to the next iter
				switch (this->layer_param_.data_param().backend()) {
				case DataParameter_DB_LEVELDB:
				  iter_->Next();
				  if (!iter_->Valid()) {
				    // We have reached the end. Restart from the first.
				    DLOG(INFO) << "Restarting data prefetching from start.";
				    iter_->SeekToFirst();
				  }
				  break;
				case DataParameter_DB_LMDB:
				  if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
				          &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
				    // We have reached the end. Restart from the first.
				    DLOG(INFO) << "Restarting data prefetching from start.";
				    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
				            &mdb_value_, MDB_FIRST), MDB_SUCCESS);
						//ExternData.InitialID = 0;
				    //ExternData.AddNewDataLocal(true, 0);
				  }
				  break;
				default:
				  LOG(FATAL) << "Unknown database backend";
				}
				
				continue;
			}
			else if ( top_label[item_id] != 0 )
			{
				//LOG(INFO) << "Change: " << top_label[item_id];
				top_label[item_id] = -1;
			}
    }

    // update to extern data
    // ExternData.AddNewDataGlobalWithKey( mdb_key_, top_label[item_id]);
    // ExternData.AddNewDataLocal(false, item_id);

    if (DATAUPDATE)
    {
			char a[9];
			for (int i = 0; i<8; i++)
			{
				a[i] = *((char *)mdb_key_.mv_data + i);
			}
			a[8] = 0;
			int curID = atoi(a);
			//LOG(INFO) << item_id << " " << curID << " " << top_label[item_id];
			ExternData.UpdateGlobalLabel( curID, top_label[item_id]);
			ExternData.UpdateSelectedID( item_id, curID);
    }
		//else LOG(INFO) << item_id << " " << curID << " " << top_label[item_id];

	
    
    // go to the next iter
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      iter_->Next();
      if (!iter_->Valid()) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        iter_->SeekToFirst();
      }
      break;
    case DataParameter_DB_LMDB:
      if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                &mdb_value_, MDB_FIRST), MDB_SUCCESS);
				//ExternData.InitialID = 0;
        //ExternData.AddNewDataLocal(true, 0);
      }
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }
  }
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
//debug
//  if (ExternData.ACTIVATE)
//  {
//	  LOG(INFO) << "The all keys obtained:";
//	  for (int i = 0; i<ExternData.NumberData; i++)
//	  {
//		  LOG(INFO) << (char*)(ExternData.Key[i].mv_data);
//	  }
//	  LOG(INFO) << "The current keys obtained:";
//	  for (int i = 0; i<ExternData.SelectedDataID.size(); i++)
//	  {
//		  LOG(INFO) << (char*)(ExternData.Key[ExternData.SelectedDataID[i]].mv_data);
//	  }
//  }

}*/

INSTANTIATE_CLASS(DataLayer);

}  // namespace caffe
