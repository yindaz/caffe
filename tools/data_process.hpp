#ifndef DATA_PROCESS_
#define DATA_PROCESS_

#include <string>
#include <vector>
#include "boost/scoped_ptr.hpp"
#include "hdf5.h"
#include "leveldb/db.h"
#include "lmdb.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

//template <typename Dtype>
typedef std::pair<double, int> mypair;
bool comparator ( const mypair& l, const mypair& r)
   { return l.first < r.first; }

template <typename Dtype>
class DataProcess
{
public:
	DataProcess();
	// DataProcess( float a, float b);
	// DataProcess( std::vector<float> &a, std::vector<float> &b);

	void ComputeMinibatchWeight(float* a, int M, int N);
	void RandomDrawFromUniverse(int N, std::vector<int> &Index);

	void SetupLMDB( const MDB_dbi &dbi, const MDB_txn &txn);
	void SetupDataNumber( const MDB_env* &env_);

	void AddNewDataGlobal( const MDB_val &key, int l);
	void AddNewDataLocal( bool reset, int id);
	void ResetDataLocal( int batchsize);


	void FindOutlier( Dtype* top_diff, Dtype* bottom_data, int M, int N, int K);
	void ComputeReweighting( Dtype* top_diff, int M);


	bool NumberInitialed; //initial number and allocate memory
	bool LabelInitialed; 
	int NumberData;
	vector<float> Weight;
	vector<float> Label;
	vector<bool> IsNegtive;
	int InitialID;

	vector<float> MiniBatchWeight;
	vector<float> MiniBatchLabel;
	vector<float> MiniBatchIsneg;
	vector<int> SelectedDataID;
	int CurrentID;
	bool IdentifyOutlier;
	vector<bool> IsOutlier;

	bool ReWeighted;

	
	MDB_dbi mdb_dbi_;
  	MDB_txn* mdb_txn_;

  	Dtype PosReweight, NegReweight;
  	int NumPos, NumNeg, NumAll, NumNull;

  	bool ACTIVATE;

};

extern DataProcess ExternData;
