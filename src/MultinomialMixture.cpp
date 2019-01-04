#include <Rcpp.h>
#include <RcppEigen.h>
#include "pottsMultinomial.h"
#include <cstdlib>


// [[Rcpp::export]]
Rcpp::List estimateMixture(Eigen::MatrixXi Y_,
                                             int k_,
                                             int nIter_,
                                             int seed_,
                                             double alpha_){
  //gamma is the regularization parameter
  MultinomialMixture mixture(Y_,k_,seed_,alpha_);
  return(mixture.solve(nIter_));
}

// [[Rcpp::export]]
Rcpp::List estimateMixtureWithErrors(Eigen::MatrixXi Y_,
                                     Eigen::MatrixXi Y_test,
                           int k_,
                           int nIter_,
                           int seed_,
                           double alpha_){
  //gamma is the regularization parameter
  MultinomialMixture mixture(Y_,k_,seed_,alpha_);
  return(mixture.solve(nIter_,Y_,Y_test));
}

// [[Rcpp::export]]
Rcpp::List estimateInitializedMixture(Eigen::MatrixXi Y_,
                           int k_,
                           int nIter_,
                           int seed_,
                           double alpha_,
                           Eigen::MatrixXd initialPYZ_,
                           Eigen::VectorXd initialH_){
  //gamma is the regularization parameter
  MultinomialMixture mixture(Y_,k_,seed_,alpha_);
  mixture.pyz=initialPYZ_;
  mixture.h=initialH_;
  return(mixture.solve(nIter_));
}

// [[Rcpp::export]]
Rcpp::List estimateInitializedMixturewithErrors(Eigen::MatrixXi Y_,
                                                Eigen::MatrixXi Y_test,
                                      int k_,
                                      int nIter_,
                                      int seed_,
                                      double alpha_,
                                      Eigen::MatrixXd initialPYZ_,
                                      Eigen::VectorXd initialH_){
  //gamma is the regularization parameter
  MultinomialMixture mixture(Y_,k_,seed_,alpha_);
  mixture.pyz=initialPYZ_;
  mixture.h=initialH_;
  return(mixture.solve(nIter_,Y_,Y_test));
}


// 
// // [[Rcpp::export]]
// std::vector<Eigen::MatrixXd> estimateInitializedMixture(Eigen::MatrixXi Y_, 
//                                              int k_, 
//                                              int nIter_, 
//                                              Eigen::MatrixXd initialTheta_,
//                                              Eigen::VectorXd initialH_,
//                                              int seed_,
//                                              double alpha_){
//   MultinomialMixture mixture(Y_,k_,initialH_,initialTheta_,seed_,alpha_);
//   mixture.solve(nIter_);
//   
//   std::vector<Eigen::MatrixXd> parList;
//   parList.push_back(mixture.logpyz.array().exp());
//   parList.push_back(mixture.h.array().exp());
//   return parList;
// }