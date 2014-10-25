#ifndef DATA_PROCESS_
#define DATA_PROCESS_

#include <string>
#include <vector>
#include <fstream>

#include "boost/scoped_ptr.hpp"
#include "hdf5.h"
#include "leveldb/db.h"
#include "lmdb.h"

#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"


using namespace std;
using namespace caffe;
//template <typename Dtype>



class DataProcess
{
public:
	DataProcess();
	// DataProcess( float a, float b);
	// DataProcess( std::vector<float> &a, std::vector<float> &b);
	typedef std::pair<double, int> mypair;
//	void ComputeMinibatchWeight(float* a, int M, int N);
//	void RandomDrawFromUniverse(int N, std::vector<int> &Index);
//
	void SetupLMDB(  MDB_dbi &dbi,  MDB_txn* &txn);
	void SetupDataNumber(  MDB_env* &env_);

	void AddNewDataGlobal( const MDB_val &key, int l);
	void AddNewDataGlobalWithKey( const MDB_val &key, int l);
	void AddNewDataLocal( bool reset, int id);
	void MoveNewDataLocal( );

	void ResetDataLocal( int batchsize);

//	void FindOutlier( float* top_diff, float* bottom_data, int &M, int &N, int &K);
//	void ComputeReweighting( float* top_diff, int M);

//	template <typename Dtype>
//	void FindOutlier( const Dtype* top_diff, const Dtype* bottom_data, int &M, int &N, int &K);

	template <typename Dtype>
	void FindOutlier( const Dtype* top_diff, const Dtype* bottom_data, int &M, int &N, int &K)
	{
		if (!ACTIVATE) return;

//		ofstream F_top_diff("top_diff.txt");
//		ofstream F_bottom_data("bottom_data.txt");
//		ofstream F_gradient("gradient.txt");
//		ofstream F_mean_gradient("mean_gradient.txt");
//		ofstream F_abs_diff("abs_diff.txt");


//		for (int i = 0; i<N; i++)
//		{
//			for (int j = 0; j<M; j++)
//			{
//				F_top_diff << *(top_diff+i*M+j) << " ";
//			}
//			F_top_diff << endl;
//		}
//		F_top_diff.close();
//
//		for (int i = 0; i<N; i++)
//		{
//			for (int j = 0; j<K; j++)
//			{
//				F_bottom_data << *(bottom_data+i*K+j) << " ";
//			}
//			F_bottom_data << endl;
//		}
//		F_bottom_data.close();



		if ( !IdentifyOutlier )
		{
			Blob<Dtype> GradMean;
	  		Blob<Dtype> Grad;

			Grad.Reshape( N, 1, M, K);
			GradMean.Reshape(1, 1, M, K);

			int top_offset = M;
			int bottom_offset = K;
			int grad_offset = M*K;

			Dtype* AllGrad = Grad.mutable_cpu_data();

			for (int i = 0; i < N; ++i)
			{
				caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M, K, 1, (Dtype)1.,
					top_diff + top_offset * i, bottom_data + bottom_offset * i, (Dtype)0.,
					AllGrad + grad_offset * i);
			}

//			for (int i = 0; i<N; i++)
//			{
//				for (int j = 0; j<grad_offset; j++)
//				{
//					F_gradient << *(AllGrad+i*grad_offset+j) << " ";
//				}
//				F_gradient << endl;
//			}
//			F_gradient.close();

			const Dtype* element_r = Grad.cpu_data();
			Dtype* eleMean_w = GradMean.mutable_cpu_data();
			caffe_set(grad_offset, (Dtype)0., eleMean_w);

			int PositiveNum = 0;
			for (int i = 0; i < N; ++i)
			{
				if (!MiniBatchIsneg[i])
				{
					caffe_axpy<Dtype>(grad_offset, (Dtype)1., element_r+i*grad_offset, eleMean_w);
					PositiveNum ++;
				}
			}
			caffe_scal<Dtype>( grad_offset, (Dtype)(1.0/PositiveNum), eleMean_w);

//			for (int i = 0; i<1; i++)
//			{
//				for (int j = 0; j<grad_offset; j++)
//				{
//					F_mean_gradient << *(eleMean_w+i*grad_offset+j) << " ";
//				}
//				F_mean_gradient << endl;
//			}
//			F_mean_gradient.close();

			Dtype* element_w = Grad.mutable_cpu_data();
			const Dtype* eleMean_r = GradMean.cpu_data();

			for (int i = 0; i < N; ++i)
			{
				if (!MiniBatchIsneg[i])
				{
					caffe_axpy<Dtype>(grad_offset, (Dtype)(-1.), eleMean_r, element_w+i*grad_offset);
					caffe_abs<Dtype>(grad_offset, Grad.cpu_data() + i * grad_offset, element_w + i*grad_offset);
				}

			}

//			for (int i = 0; i<N; i++)
//			{
//				for (int j = 0; j<grad_offset; j++)
//				{
//					F_abs_diff << *(element_w+i*grad_offset+j) << " ";
//				}
//				F_abs_diff << endl;
//			}
//			F_abs_diff.close();

			Blob<Dtype> Ones;
			Ones.Reshape( 1, 1, grad_offset, 1);
			caffe_set(grad_offset, (Dtype)1., Ones.mutable_cpu_data());

			Blob<Dtype> OutIndicator;
			OutIndicator.Reshape( N, 1, 1, 1);
			caffe_cpu_gemv<Dtype>(CblasNoTrans, N, grad_offset, (Dtype) 1., Grad.cpu_data(), Ones.cpu_data(),
	    	(Dtype)0., OutIndicator.mutable_cpu_data());

			const Dtype* diff = OutIndicator.cpu_data();
			std::vector<mypair> scores;
			scores.clear();
			for (int i = 0; i < N; ++i)
			{
				if (!MiniBatchIsneg[i])
				{
					scores.push_back(mypair(*(diff+i), i));
				}
			}
			std::sort(scores.begin(), scores.end(), DataProcess::comparator_pair_index);

			int KillNumber = (int) (PositiveNum * 0);
			IsOutlier.assign(N, false);
			for (int i = 0; i < KillNumber; ++i)
			{
				IsOutlier[scores[PositiveNum-i-1].second] = true;
				Weight[MiniBatchDataID[scores[PositiveNum-i-1].second]]++;
			}

			IdentifyOutlier = true;
		}
	};

//	template <typename Dtype>
//	void ComputeReweighting( Dtype* top_diff, int M);

	template <typename Dtype>
	void ComputeReweighting(Dtype* top_diff, int M)
	{
		if (!ACTIVATE) return;
		if ( !ReWeighted )
		{
			NumAll = MiniBatchLabel.size();
			NumPos = 0;
			NumNeg = 0;
			NumNull = 0;

			for (int i = 0; i < NumAll; ++i)
			{
				if (IsOutlier[i])
				{
					NumNull ++;
				}
				else
				{
					if (MiniBatchIsneg[i])
					{
						NumNeg ++;
					}
					else
					{
						NumPos ++;
					}
				}
			}

			PosReweight = (Dtype)1. / (Dtype)2. / (Dtype)NumPos;
			NegReweight = (Dtype)1. / (Dtype)2. / (Dtype)NumNeg;

			for (int i = 0; i < NumAll; ++i)
			{
				if (IsOutlier[i])
				{
					caffe_scal<Dtype>( M, (Dtype)(0.), top_diff + i * M);
				}
				else
				{
					if ( !MiniBatchIsneg[i] )
					{
						caffe_scal<Dtype>( M, (Dtype)(PosReweight), top_diff + i * M);
					}
					else
					{
						caffe_scal<Dtype>( M, (Dtype)(PosReweight), top_diff + i * M);
					}
				}
			}
			ReWeighted = true;
		}
	}



	static bool comparator_pair_index ( const mypair& l, const mypair& r)
	   { return l.first < r.first; }

	bool NumberInitialed; //initial number and allocate memory
	bool LabelInitialed; 
	int NumberData;
	vector<float> Weight;
	vector<float> Label;
	vector<MDB_val> Key;
	vector<bool> IsNegative;
	int InitialID;

	vector<float> MiniBatchWeight;
	vector<float> MiniBatchLabel;
	vector<bool> MiniBatchIsneg;
	vector<int> SelectedDataID;
	vector<int> MiniBatchDataID;
	bool MiniBatchUpdated;
	int CurrentID;
	bool IdentifyOutlier;
	vector<bool> IsOutlier;

	bool ReWeighted;

	
	MDB_dbi mdb_dbi_;
  	MDB_txn* mdb_txn_;

  	float PosReweight, NegReweight;
  	int NumPos, NumNeg, NumAll, NumNull;

  	bool ACTIVATE;

};


#endif
