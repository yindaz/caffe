#include <caffe/data_process.hpp>
#include <string>
#include <algorithm>
#include "caffe/blob.hpp"

using namespace caffe;

DataProcess::DataProcess()
{
	ACTIVATE = true;

	NumberInitialed = false;
	LabelInitialed = false;
	InitialID = 0;
	CurrentID = 0;
	NumberData = 0;

	NoiseRate = 0;
}

void DataProcess::SetupLMDB(  MDB_dbi &dbi,  MDB_txn* &txn)
{
	if (!ACTIVATE) return;

	mdb_dbi_ = dbi;
	mdb_txn_ = txn;
}

void DataProcess::SetupDataNumber( MDB_env* &env_)
{
	if (!ACTIVATE) return;

	if ( !NumberInitialed )
	{
		MDB_stat stat;
		mdb_env_stat( env_, &stat);
		NumberData = stat.ms_entries;

		Weight.assign(NumberData, 0);
		Label.assign(NumberData, 0);
		Key.assign(NumberData, MDB_val());
		IsNegative.assign(NumberData, false);

		NumberInitialed = true;
	}

}

void DataProcess::AddNewDataGlobal( const MDB_val &key, int l)
{
	if (!ACTIVATE) return;

	if ( InitialID<NumberData)
	{
		string s(reinterpret_cast<char*>(key.mv_data));
		int j = InitialID; //std::stoi(s);
		if ( j != InitialID )
		{
			LOG(FATAL)<<"ID mismatch!";
		}
		if (l>=0)
		{
			Label[InitialID] = l;
			IsNegative[InitialID] = false;
		}
		else
		{
			Label[InitialID] = -l - 1;
			IsNegative[InitialID] = true;
		}
		InitialID++;
	}
}

void DataProcess::AddNewDataGlobalWithKey( const MDB_val &key, int l)
{
	if (!ACTIVATE) return;

	if ( InitialID<NumberData)
	{
		string s(reinterpret_cast<char*>(key.mv_data));
		int j = InitialID; //std::stoi(s);
		if ( j != InitialID )
		{
			LOG(FATAL)<<"ID mismatch!";
		}
		if (l>=0)
		{
			Label[InitialID] = l;
			IsNegative[InitialID] = false;
		}
		else
		{
			Label[InitialID] = -l - 1;
			IsNegative[InitialID] = true;
		}
		Key[InitialID] = key;
		InitialID++;
	}
}

void DataProcess::AddNewDataLocal( bool reset, int id)
{
	if (!ACTIVATE) return;

	if (reset)
	{
		CurrentID = id;
	}
	else
	{
		SelectedDataID[id] = CurrentID;
//		MiniBatchWeight[id] = Weight[CurrentID];
//		MiniBatchLabel[id] = Label[CurrentID];
//		MiniBatchIsneg[id] = IsNegative[CurrentID];
		CurrentID ++;
	}
	MiniBatchUpdated = false;
}

void DataProcess::MoveNewDataLocal( )
{
	for ( int i = 0; i<SelectedDataID.size(); ++i)
	{
		MiniBatchWeight[i] = Weight[SelectedDataID[i]];
		MiniBatchLabel[i] = Label[SelectedDataID[i]];
		MiniBatchIsneg[i] = IsNegative[SelectedDataID[i]];
		MiniBatchDataID[i] = SelectedDataID[i];
	}
	MiniBatchUpdated = true;
}

void DataProcess::ResetDataLocal( int batchsize)
{
	if (!ACTIVATE) return;

	//SelectedDataID.assign(batchsize, -1);
	MiniBatchWeight.assign(batchsize, -1);
	MiniBatchLabel.assign(batchsize, -1);
	MiniBatchIsneg.assign(batchsize, -1);
	MiniBatchDataID.assign(batchsize, -1);
	IdentifyOutlier = false;
	IsOutlier.assign(0, true);
	ReWeighted = false;
}

//void DataProcess::FindOutlier( float* top_diff, float* bottom_data, int &M, int &N, int &K)
//{
//	if (!ACTIVATE) return;
//
//	if ( !IdentifyOutlier )
//	{
//		Blob<Dtype> GradMean;
//  		Blob<Dtype> Grad;
//
//		Grad.Reshape( N, 1, M, K);
//		GradMean.Reshape(1, 1, M, K);
//
//		int top_offset = M;
//		int bottom_offset = K;
//		int grad_offset = M*K;
//
//		Dtype* AllGrad = Grad.mutable_cpu_data();
//
//		for (int i = 0; i < N; ++i)
//		{
//			caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M, K, 1, (Dtype)1.,
//				top_diff + top_offset * i, bottom_data + bottom_offset * i, (Dtype)0.,
//				AllGrad + grad_offset * i);
//		}
//
//
//		Dtype* element = Grad.cpu_data();
//		Dtype* eleMean = GradMean.mutable_cpu_data();
//		caffe_set(grad_offset, (Dtype)0., eleMean);
//
//		for (int i = 0; i < N; ++i)
//		{
//			caffe_axpy<Dtype>(grad_offset, (Dtype)1., element+N*grad_offset, eleMean);
//		}
//		caffe_scal<Dtype>( grad_offset, (Dtype)(1.0/N), eleMean);
//
//		element = Grad.mutable_cpu_data();
//		eleMean = GradMean.cpu_data();
//
//		for (int i = 0; i < N; ++i)
//		{
//			caffe_axpy<Dtype>(grad_offset, (Dtype)(-1.), eleMean, element+N*grad_offset);
//			caffe_abs<Dtype>(grad_offset, Grad.cpu_data() + N * grad_offset, element + N*grad_offset);
//		}
//
//		Blob<Dtype> Ones;
//		Ones.Reshape( 1, 1, grad_offset, 1);
//		caffe_set(grad_offset, (Dtype)1., Ones.mutable_cpu_data());
//
//		Blob<Dtype> OutIndicator;
//		OutIndicator.Reshape( N, 1, 1, 1);
//		caffe_cpu_gemv<float>(CblasNoTrans, N, grad_offset, (Dtype) 1., Grad.cpu_data(), Ones.cpu_data(),
//    	(Dtype)0., OutIndicator.mutable_cpu_data());
//
//		Dtype* diff = OutIndicator.cpu_data();
//		std::vector<mypair> scores;
//		scores.clear();
//		for (int i = 0; i < N; ++i)
//		{
//			scores.push_back(mypair(*(diff+i), i));
//		}
//		std::sort(scores.begin(), scores.end(), DataProcess::comparator_pair_index);
//
//		int KillNumber = (int) (N * 0.05);
//		IsOutlier.assign(N, false);
//		for (int i = 0; i < KillNumber; ++i)
//		{
//			IsOutlier[scores[N-i-1].second] = true;
//		}
//
//		IdentifyOutlier = true;
//	}
//}
//
//template <typename Dtype>
//void DataProcess::ComputeReweighting(Dtype* top_diff, int M)
//{
//	if (!ACTIVATE) return;
//	if ( !ReWeighted )
//	{
//		NumAll = MiniBatchLabel.size();
//		NumPos = 0;
//		NumNeg = 0;
//		NumNull = 0;
//
//		for (int i = 0; i < NumAll; ++i)
//		{
//			if (IsOutlier[i])
//			{
//				NumNull ++;
//			}
//			else
//			{
//				if (MiniBatchIsneg[i])
//				{
//					NumNeg ++;
//				}
//				else
//				{
//					NumPos ++;
//				}
//			}
//		}
//
//		PosReweight = (Dtype)NumAll / (Dtype)2. / (Dtype)NumPos;
//		NegReweight = (Dtype)NumAll / (Dtype)2. / (Dtype)NumNeg;
//
//		for (int i = 0; i < NumAll; ++i)
//		{
//			if (IsOutlier[i])
//			{
//				caffe_scal<Dtype>( top_diff + i * M, (Dtype)(0.), top_diff + i * M);
//			}
//			else
//			{
//				if ( !IsNegative[i] )
//				{
//					caffe_scal<Dtype>( top_diff + i * M, (Dtype)(PosReweight), top_diff + i * M);
//				}
//				else
//				{
//					caffe_scal<Dtype>( top_diff + i * M, (Dtype)(PosReweight), top_diff + i * M);
//				}
//			}
//		}
//		ReWeighted = true;
//	}
//}


/*template <typename Dtype>
void DataProcess::FindOutlier( const Dtype* top_diff, const Dtype* bottom_data, int &M, int &N, int &K)
{
	if (!ACTIVATE) return;

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


		Dtype* element = Grad.cpu_data();
		Dtype* eleMean = GradMean.mutable_cpu_data();
		caffe_set(grad_offset, (Dtype)0., eleMean);

		for (int i = 0; i < N; ++i)
		{
			caffe_axpy<Dtype>(grad_offset, (Dtype)1., element+N*grad_offset, eleMean);
		}
		caffe_scal<Dtype>( grad_offset, (Dtype)(1.0/N), eleMean);

		element = Grad.mutable_cpu_data();
		eleMean = GradMean.cpu_data();

		for (int i = 0; i < N; ++i)
		{
			caffe_axpy<Dtype>(grad_offset, (Dtype)(-1.), eleMean, element+N*grad_offset);
			caffe_abs<Dtype>(grad_offset, Grad.cpu_data() + N * grad_offset, element + N*grad_offset);
		}

		Blob<Dtype> Ones;
		Ones.Reshape( 1, 1, grad_offset, 1);
		caffe_set(grad_offset, (Dtype)1., Ones.mutable_cpu_data());

		Blob<Dtype> OutIndicator;
		OutIndicator.Reshape( N, 1, 1, 1);
		caffe_cpu_gemv<float>(CblasNoTrans, N, grad_offset, (Dtype) 1., Grad.cpu_data(), Ones.cpu_data(),
    	(Dtype)0., OutIndicator.mutable_cpu_data());

		Dtype* diff = OutIndicator.cpu_data();
		std::vector<mypair> scores;
		scores.clear();
		for (int i = 0; i < N; ++i)
		{
			scores.push_back(mypair(*(diff+i), i));
		}
		std::sort(scores.begin(), scores.end(), DataProcess::comparator_pair_index);

		int KillNumber = (int) (N * 0.05);
		IsOutlier.assign(N, false);
		for (int i = 0; i < KillNumber; ++i)
		{
			IsOutlier[scores[N-i-1].second] = true;
		}

		IdentifyOutlier = true;
	}
}*/
//
//template <typename Dtype>
//void DataProcess::ComputeReweighting(Dtype* top_diff, int M)
//{
//	if (!ACTIVATE) return;
//	if ( !ReWeighted )
//	{
//		NumAll = MiniBatchLabel.size();
//		NumPos = 0;
//		NumNeg = 0;
//		NumNull = 0;
//
//		for (int i = 0; i < NumAll; ++i)
//		{
//			if (IsOutlier[i])
//			{
//				NumNull ++;
//			}
//			else
//			{
//				if (MiniBatchIsneg[i])
//				{
//					NumNeg ++;
//				}
//				else
//				{
//					NumPos ++;
//				}
//			}
//		}
//
//		PosReweight = (Dtype)NumAll / (Dtype)2. / (Dtype)NumPos;
//		NegReweight = (Dtype)NumAll / (Dtype)2. / (Dtype)NumNeg;
//
//		for (int i = 0; i < NumAll; ++i)
//		{
//			if (IsOutlier[i])
//			{
//				caffe_scal<Dtype>( top_diff + i * M, (Dtype)(0.), top_diff + i * M);
//			}
//			else
//			{
//				if ( !IsNegative[i] )
//				{
//					caffe_scal<Dtype>( top_diff + i * M, (Dtype)(PosReweight), top_diff + i * M);
//				}
//				else
//				{
//					caffe_scal<Dtype>( top_diff + i * M, (Dtype)(PosReweight), top_diff + i * M);
//				}
//			}
//		}
//		ReWeighted = true;
//	}
//}




































