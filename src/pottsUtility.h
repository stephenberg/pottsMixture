#include <Rcpp.h>
#include <RcppEigen.h>
#include "graph.h"
#include <cstdlib>

using namespace Rcpp;
using Rcpp::Rcout;

typedef Eigen::MatrixXi intMat;
typedef Eigen::VectorXi intVec;
typedef Eigen::MatrixXd matrix;
typedef Eigen::VectorXd vec;

class pottsUtility{
protected:


public:
  int nCategory;
  int n;
  
  int nGraph;
  graph edgeList;
  
  intVec graphMembershipCounts;
  std::vector<intVec> graphMemberships;
  
  bool offDiagonal;
  matrix localFields;
  pottsUtility(int n_, int nCategory_, intMat edgeList_, intVec edgeStart_, intVec edgeEnd_):edgeList(edgeList_,edgeStart_,edgeEnd_)
{
    n=n_;
    nCategory=nCategory_;
};
  
  
  std::vector<intMat> computeSufficientStatistics(intVec& z){
    intMat prevalenceStatistic;
    intMat correlationStatistic;
    
    prevalenceStatistic.setZero(nCategory,1);
    correlationStatistic.setZero(nCategory,nCategory);
    
    for (int i=0;i<n;i++){
      //compute prevalence statistics
      prevalenceStatistic(z(i),0)+=1;
      for (int i1=0;i1<=edgeList.nNeighbors(i);i1++){
        int iPrime=edgeList.getNeighbor(i,i1);
        if (iPrime>i){
          correlationStatistic(z(i),z(iPrime))+=1;
          correlationStatistic(z(iPrime),z(i))+=1;
        }
      }
    }
    std::vector<intMat> sufficientStatistic;
    sufficientStatistic.push_back(prevalenceStatistic);
    sufficientStatistic.push_back(correlationStatistic);
    return sufficientStatistic;
  };


  vec computeLocalField(int position,
                         matrix& field,
                         matrix& correlations,
                         intVec& z){

    vec currentField;
    currentField.setZero(nCategory);

    for (int kInd=0;kInd<nCategory;kInd++){
      currentField(kInd)=field(position,kInd);
      for (int i1=0;i1<=edgeList.nNeighbors(position);i1++){
        int iPrime=edgeList.getNeighbor(position,i1);
        if (iPrime!=position){
          currentField(kInd)+=correlations(kInd,z(iPrime));
        }
      }
    }
    return currentField;
  }


  double runif(){
    return((double) rand() / (RAND_MAX));
  }

  int drawMultinomial(vec& probabilities)
  {
    double rUnif=(double) rand() / (RAND_MAX);
    double k=probabilities.size();
    double sum=0;
    for (int kInd=0;kInd<k;kInd++){
      sum=sum+probabilities(kInd);
      if (rUnif<sum){
        return(kInd);
      }
    }
    return(0);
  }
  
  //gibbs sampling and conditional sufficient statistic functions
  //interface some utility functions with the pottsHess functions
  std::vector<intVec> gibbs_Sample(intVec& start,
                                            int nIterations,
                                            bool fullPath,
                                            matrix& field,
                                            matrix& correlations){
    
    Eigen::VectorXi configuration=start;
    Eigen::VectorXd currentProbabilities;
    currentProbabilities.setZero(nCategory);
    
    vec currentField;
    
    std::vector<Eigen::VectorXi> configList;
    if (fullPath){
      configList.push_back(configuration);
    }
    for (int iter=0;iter<nIterations;iter++){
      for (int nInd=0;nInd<n;nInd++){
        
        
        //////////////////////get local field at location. Safely convert to probability.
        currentField=computeLocalField(nInd,field,correlations,configuration);
        double maxField=currentField.maxCoeff();
        currentField=currentField.array()-maxField;
        double expSum=currentField.array().exp().sum();
        currentProbabilities=currentField.array().exp()/expSum;
        /////////////////////
        
        //sample
        configuration(nInd)=drawMultinomial(currentProbabilities);
      }
      if (fullPath){
        configList.push_back(configuration);
      }
    }
    //only push back the last configuration
    if (!fullPath){
      configList.push_back(configuration);
    }
    return(configList);
  };
  
};