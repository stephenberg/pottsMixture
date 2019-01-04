// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
//#include "pottsHessian.h"
#include <cstdlib>
#include "pottsMLE.h"
using Rcpp::Rcout;

//Gibbs sampling wrapper function
//[[Rcpp::export]]
std::vector<Eigen::VectorXi> gibbsSample(int k_,
                                        Eigen::MatrixXi edge1_,
                                        Eigen::VectorXi start1_,
                                        Eigen::VectorXi end1_,
                                        int seed_,
                                        int nIter_,
                                        bool offDiagonal_,
                                        bool fullPath_,
                                        Eigen::MatrixXd h_,
                                        Eigen::MatrixXd correlations_){
  int n=start1_.size();
  pottsUtility sampler(n,
                        k_,
                        edge1_,
                        start1_,
                        end1_);

  std::srand(seed_);
  intVec z;
  z.setZero(n);
  matrix field;
  field.setZero(n,k_);
  matrix ones;
  ones.setOnes(n,1);
  field=ones*h_.transpose();
  for (int i=0;i<n;i++){
    z(i)=i%k_;
  }
  std::vector<intVec> configList=sampler.gibbs_Sample(z,nIter_,fullPath_,field,correlations_);
  return(configList);
}

//unknown noise stochastic approximation EM without initialization
//[[Rcpp::export]]
Rcpp::List EMSA(Eigen::MatrixXi Y_,
                int k_,
                Eigen::MatrixXi edge1_,
                Eigen::VectorXi start1_,
                Eigen::VectorXi end1_,
                int seed_,
                int constantIterations_,
                int decreasingIterations_,
                int decreasingStart_,
                bool scalarCorrelations_,
                double stepSize_,
                double alpha_,
                bool stochasticGradient_,
                bool surrogate_){

  pottsMLE mleEstimator(Y_,
                        k_,
                        edge1_,
                        start1_,
                        end1_,
                        seed_,
                        alpha_);
  return mleEstimator.EM_SA(constantIterations_,decreasingIterations_,decreasingStart_,stepSize_,stochasticGradient_,surrogate_,scalarCorrelations_);
}


//unknown noise stochastic approximation EM without initialization
//[[Rcpp::export]]
Rcpp::List initializedEMSA(Eigen::MatrixXi Y_,
                int k_,
                Eigen::MatrixXi edge1_,
                Eigen::VectorXi start1_,
                Eigen::VectorXi end1_,
                int seed_,
                int constantIterations_,
                int decreasingIterations_,
                int decreasingStart_,
                bool scalarCorrelations_,
                double stepSize_,
                double alpha_,
                bool stochasticGradient_,
                bool surrogate_,
                Eigen::MatrixXd initialH_,
                Eigen::MatrixXd initialCorrelations_,
                Eigen::MatrixXd initialMu_){
  
  pottsMLE mleEstimator(Y_,
                        k_,
                        edge1_,
                        start1_,
                        end1_,
                        seed_,
                        alpha_);
  mleEstimator.prevalenceParameters=initialH_;
  mleEstimator.correlationParameters=initialCorrelations_;
  mleEstimator.multMixture.pyz=initialMu_;
  return mleEstimator.EM_SA(constantIterations_,decreasingIterations_,decreasingStart_,stepSize_,stochasticGradient_,surrogate_,scalarCorrelations_);
}


// 
// 
// //posterior predictive CV
//[[Rcpp::export]]
Rcpp::List cv_posteriorPredictive(Eigen::MatrixXi YTrain_,
                                  Eigen::MatrixXi YTest_,
                                  Eigen::MatrixXi edge1_,
                                  Eigen::VectorXi start1_,
                                  Eigen::VectorXi end1_,
                                  int seed_,
                                  int nIter_,
                                  int nBurn_,
                                  Eigen::MatrixXd H_,
                                  Eigen::MatrixXd muParameter_,
                                  Eigen::MatrixXd correlations_){
  int k_=muParameter_.cols();

  pottsMLE mleEstimator(YTrain_,
                        k_,
                        edge1_,
                        start1_,
                        end1_,
                        seed_,
                        2);

  mleEstimator.multMixture.pyz=muParameter_;
  mleEstimator.correlationParameters=correlations_;
  mleEstimator.prevalenceParameters=H_;
  // mleEstimator.crossValidate(YTrain_,nIter_,nBurn_);
  // return 0;
  return mleEstimator.crossValidate(YTest_,nIter_,nBurn_);
}
