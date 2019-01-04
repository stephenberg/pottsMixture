#include <Rcpp.h>
#include <RcppEigen.h>
#include <cstdlib>
using namespace Rcpp;
using Rcpp::Rcout;

typedef Eigen::MatrixXi intMat;
typedef Eigen::VectorXi intVec;
typedef Eigen::MatrixXd matrix;
typedef Eigen::VectorXd vec;

class graph{
protected:
  intMat edgeList;
  intVec edgeStart;
  intVec edgeEnd;
  intVec numNeighbors;
  int length;
  int n;

  
public:
  
  graph(intMat edgeList_, intVec edgeStart_, intVec edgeEnd_){
    edgeList=edgeList_;
    edgeStart=edgeStart_;
    edgeEnd=edgeEnd_;
    numNeighbors.setZero(edgeStart.size());
    length=edgeStart.size();
    for (int i=0;i<edgeStart.size();i++){
      numNeighbors(i)=edgeEnd(i)-edgeStart(i);
    }
  }
  
  int getNeighbor(int i, int whichNeighbor){
    int neighborIndex=edgeStart(i)+whichNeighbor;
    int neighbor=edgeList(neighborIndex,1);
    return(neighbor);
  }
  
  int nNeighbors(int i){
    return(numNeighbors(i));
  }
  
};

