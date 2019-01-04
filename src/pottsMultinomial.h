//#pragma once
#include <RcppEigen.h>
#include <cstdlib>
#include <stdlib.h>
#include <vector>
#include <random>

typedef Eigen::MatrixXi intMat;
typedef Eigen::VectorXi intVec;
typedef Eigen::MatrixXd matrix;
typedef Eigen::VectorXd vec;
using Eigen::Map;
using namespace Eigen;

class MultinomialMixture {
public:
  //number of locations
  int n;

  //number of tree species
  int nSpecies;

  //number of trees in the sample
  int treeCount;
  
  //error matrix
  matrix errorMatrix;

  //number of landscape type categories
  int k;

  // m-by-k matrix with entry p,q = (p(yi=p|zi=q))
  matrix pyz;


  //regularization parameter for Dirichlet prior on the entries of p(y|z)
  //alpha=1 gives no regularization
  double alpha;


  //p by k coefficient matrix
  matrix beta;

  int p;

  //k vector with entry p=log(p(zi=p))
  vec h;


  // observed tree species (nTrees by 1)
  intVec y;
  intVec yCounts;

  //y[locationStart[i]]:y[locationEnd[i]] are the observations corresponding to
  //location i
  intVec locationStart;
  intVec locationEnd;

  //n-by-k matrix of conditional probabilities given current h and pyz
  matrix condProbs;
  matrix condPotential;

  MultinomialMixture(intMat Y_, int k_, int seed,double alpha_) {
    alpha=alpha_;
    std::srand(seed);
    treeCount=Y_.array().sum();
    n = Y_.rows();
    nSpecies = Y_.cols();
    k = k_;
    h = VectorXd::Random(k);
    pyz = MatrixXd::Random(nSpecies, k);
    pyz = pyz.array().exp();
    for (int i=0;i<k;i++){
      pyz.col(i)=pyz.col(i)/pyz.col(i).array().sum();
    }

    int nUniqueTrees = (Y_.array() != 0).count();
    y.setZero(nUniqueTrees);
    yCounts.setZero(nUniqueTrees);
    condProbs.setZero(n, k);
    condPotential = condProbs;
    locationStart.setZero(n, 1);
    locationEnd.setZero(n, 1);

    int uniqueTreeIndex = 0;


    for (int location = 0; location<n; location++) {
      if (location == 0) {
        locationStart(location) = 0;
        locationEnd(location) = -1;
      }
      else {
        locationStart(location) = locationEnd(location - 1) + 1;
        locationEnd(location) = locationStart(location) - 1;
      }

      //some of the grid cells may be empty
      bool anyTrees;
      for (int species = 0; species<nSpecies; species++) {
        if (Y_(location, species) != 0) {
          anyTrees=true;
          y(uniqueTreeIndex) = species;
          yCounts(uniqueTreeIndex) = Y_(location, species);
          uniqueTreeIndex = uniqueTreeIndex + 1;
          locationEnd(location) += 1;
        }
      }

      //if no trees at this cell, add a tree species with zero counts
      //so that the cell is still included
      if (!anyTrees){
        y(uniqueTreeIndex)=0;
        yCounts(uniqueTreeIndex)=0;
        uniqueTreeIndex+=1;
        locationEnd(location)+=1;
      }
    }
  }

  Eigen::Map<Eigen::VectorXi> getYValues(int location) {
    int start = locationStart(location);
    int end = locationEnd(location);
    int groupLength = end - start + 1;
    return(Map<Eigen::VectorXi>(&y(start), groupLength,1));
  }

  Eigen::Map<Eigen::VectorXi> getYCounts(int location) {

    int start = locationStart(location);
    int end = locationEnd(location);
    int groupLength = end - start + 1;
    return(Map<Eigen::VectorXi>(&(yCounts(start)), groupLength,1));
  }

  double log_pYandZ(int location, int z) {
    double val = 0;
    Eigen::Map<VectorXi> yValues_map = getYValues(location);
    Eigen::Map<VectorXi> yCounts_map = getYCounts(location);

    for (int i = 0; i<yValues_map.rows(); i++) {
      val = val + std::log(pyz(yValues_map(i), z))*yCounts_map(i);
    }

    condPotential(location, z) = val;
    val = val + h(z);
    return(val);
  }
  

  //update condPotential
  void computeCondField() {
    for (int location = 0; location<n; location++) {
      for (int category = 0; category<k; category++) {
        log_pYandZ(location, category);
      }
    }
  }
  
  void computeCondProbs() {

    condProbs.setZero(n, k);
    for (int location = 0; location<n; location++) {
      double rowMax = 0;
      for (int category = 0; category<k; category++) {
        double lpyz = log_pYandZ(location, category);
        condProbs(location, category) = lpyz;
        if (lpyz>rowMax) {
          rowMax = lpyz;
        }
      }

      for (int category = 0; category<k; category++) {
        condProbs(location, category) -= rowMax;
      }
    }
    condProbs = condProbs.array().exp();
    Eigen::VectorXd rowSums;
    rowSums = condProbs.rowwise().sum();
    condProbs = rowSums.asDiagonal().inverse()*condProbs;
  }

  void updateParameters() {
    computeCondProbs();
    h.segment(0,k)=condProbs.colwise().sum().array()+(alpha-1);
    h.segment(0,k)=h.segment(0,k)/(h.segment(0,k).array().sum());
    h.segment(0,k)=h.segment(0,k).array().log();


    Eigen::VectorXd normalizer;
    normalizer.setZero(k);
    pyz.setZero(pyz.rows(),pyz.cols());
    
    for (int location = 0; location < n; location++) {
      Eigen::Map<VectorXi> yValues_map = getYValues(location);
      Eigen::Map<VectorXi> yCounts_map = getYCounts(location);
      for (int category = 0; category < k; category++) {
        for (int treeIndex = 0; treeIndex < yValues_map.rows(); treeIndex++) {
          pyz(yValues_map(treeIndex), category) += condProbs(location, category)*yCounts_map(treeIndex);
        }
      }
    }

    //add in regularization parameter
    pyz=pyz.array()+(alpha-1);
    normalizer=pyz.colwise().sum();
    pyz = pyz*(normalizer.asDiagonal().inverse());
  }

  Rcpp::List solve(int nIter) {
    vec hStart=h;
    matrix pyzStart=pyz;
    for (int iter = 0; iter < nIter; iter++) {
      if (iter%100==0){
        std::cout << "Iteration "<<iter << "\n";  
      }
      
      updateParameters();
    }

    // std::vector<matrix> output;
    // output.push_back(h);
    // output.push_back(pyz);
    // output.push_back(condProbs);
    // output.push_back(condPotential);
    return Rcpp::List::create(Rcpp::Named("hStart")=hStart,
                              Rcpp::Named("pyzStart")=pyzStart,
                              Rcpp::Named("h")=h,
                              Rcpp::Named("pyz")=pyz,
                              Rcpp::Named("condProbs")=condProbs);
  }
  
  Rcpp::List solve(int nIter,intMat& YTrain,intMat& YTest) {
    vec hStart=h;
    matrix pyzStart=pyz;
    matrix errors;
    errors.setZero(3,nIter);
    
    for (int iter = 0; iter < nIter; iter++) {
      std::cout<<iter<<std::endl;
      if (iter%100==0){
        std::cout << "Iteration "<<iter << "\n";  
      }
      errors.col(iter)=errorMeasures(YTrain,YTest);
      updateParameters();
    }
  
    return Rcpp::List::create(Rcpp::Named("hStart")=hStart,
                              Rcpp::Named("pyzStart")=pyzStart,
                              Rcpp::Named("h")=h,
                              Rcpp::Named("pyz")=pyz,
                              Rcpp::Named("condProbs")=condProbs,
                              Rcpp::Named("errors")=errors);
  }
  

  intVec allTrees_i(int location) {
    Eigen::Map<VectorXi> y_values_i = getYValues(location);
    Eigen::Map<VectorXi> y_counts_i = getYCounts(location);
    int n_i = y_counts_i.array().sum();
    intVec allTrees;
    allTrees.setZero(n_i);
    int slot = 0;
    for (int i = 0; i < y_values_i.size(); i++) {
      for (int j = 0; j < y_counts_i(i); j++) {
        allTrees(slot) = y_values_i(i);
        slot = slot + 1;
      }
    }
    return(allTrees);
  }

  void updateCounts(matrix& z, matrix& counts) {
    for (int i = 0; i<z.rows(); i++) {
      for (int j = 0; j<z.cols(); j++) {
        Eigen::Map<VectorXi> y_values_i = getYValues(i);
        Eigen::Map<VectorXi> y_counts_i = getYCounts(i);
        for (int slot = 0; slot<y_values_i.size(); slot++) {
          counts(y_values_i(slot), j) += (y_counts_i(slot)*z(i, j));
        }
      }
    }
  }
  
  
  vec errorMeasures(intMat& YTrain,intMat& YTest){
    vec errors;
    errors.setZero(3);
    errors(0)=marginalLogLikelihood();
    errors(1)=euclideanDistance(0,YTrain,YTest);
    errors(2)=euclideanDistance(1,YTrain,YTest);
    return(errors);
  }
  
  double marginalLogLikelihood(){
    vec probs=h;
    probs=probs.array().exp();
    probs=probs/probs.array().sum();
    h=probs.array().log();
    
    double logLike=0;
    for (int i=0;i<n;i++){
      double prob_i=0;
      for (int cat=0;cat<k;cat++){
        double prob_ik=std::exp(log_pYandZ(i,cat));
        prob_i=prob_i+prob_ik;
      }
      logLike+=std::log(prob_i);
    }
    return logLike;
  }
  
  double euclideanDistance(int setting_,intMat& YTrain_, intMat& Y_test){
    vec probs=h;
    probs=probs.array().exp();
    probs=probs/probs.array().sum();
    h=probs.array().log();
    
    computeCondProbs();
    double val=0;
    for (int i=0;i<n;i++){
      vec mu;
      mu.setZero(pyz.rows());
      if (Y_test.row(i).array().sum()>0){
        if (setting_==0){
          
          int maxCat=0;
          for (int cat=0;cat<k;cat++){
            if (condProbs(i,cat)>condProbs(i,maxCat)){
              maxCat=cat;
            }
          }
          mu=pyz.col(maxCat);
        }
        else{
          int nTrees=YTrain_.row(i).array().sum();
          double currentBest=0;
          for (int j=0;j<nSpecies;j++){
            currentBest+=std::pow(YTrain_(i,j)-nTrees*pyz(j,0),2);
          }
          int maxCat=0;
          for (int cat=0;cat<k;cat++){
            double tempBest=0;
            for (int j=0;j<nSpecies;j++){
              tempBest+=std::pow(YTrain_(i,j)-nTrees*pyz(j,cat),2);
            }
            if (tempBest<currentBest){
              currentBest=tempBest;
              maxCat=cat;
            }
          }
          mu=pyz.col(maxCat);
        }
        int nTrees=Y_test.row(i).array().sum();
        mu=mu*nTrees;
        for (int j=0;j<pyz.rows();j++){
          val+=std::pow(Y_test(i,j)-mu(j),2);
        }
      }
    }
    return val;
  }
  


  std::vector<matrix> get_pyz_components(intVec& z) {
    matrix T_condProbs;
    matrix ET_condProbs;
    T_condProbs.setZero(pyz.rows(), pyz.cols());
    ET_condProbs.setZero(pyz.rows(), pyz.cols());

    for (int i = 0; i<z.size(); i++) {
      Eigen::Map<VectorXi> y_values_i = getYValues(i);
      Eigen::Map<VectorXi> y_counts_i = getYCounts(i);
      int n_i = y_counts_i.array().sum();


      for (int slot = 0; slot<y_values_i.size(); slot++) {
        T_condProbs(y_values_i(slot), z(i)) += y_counts_i(slot);
      }
      ET_condProbs.block(0, z(i), nSpecies, 1) += n_i*pyz.block(0, z(i), nSpecies, 1);
    }

    //add regularization component: add pseudo-observations to each column of T_condProbs
    T_condProbs=T_condProbs.array()+(alpha-1);
    for (int k_=0;k_<k;k_++){
      // double sampleSize=T_condProbs.col(k_).sum();
      // double multiplier=sampleSize/(ET_condProbs.col(k_).sum());
      // ET_condProbs.col(k_)=ET_condProbs.col(k_).array()*multiplier;
      ET_condProbs.col(k_)=ET_condProbs.col(k_)+(ET_condProbs.col(k_).size()*(alpha-1))*pyz.col(k_);
    }
    //////////


    std::vector<matrix> gradComponentList;
    gradComponentList.push_back(T_condProbs);
    gradComponentList.push_back(ET_condProbs);
    return(gradComponentList);
  }
};


