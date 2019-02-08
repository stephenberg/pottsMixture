#include <Rcpp.h>
#include <RcppEigen.h>
#include <cstdlib>
#include "pottsUtility.h"
#include "pottsMultinomial.h"

class pottsMLE{
protected:



public:
  //will do computations given the state of a potts model and its parameters
  pottsUtility pottsModel;

  //stores parameters relating to the multinomial mixture model (ie the h/beta and
  //p(y|z) parameters)
  //also computes conditional potentials given observed Y
  MultinomialMixture multMixture;

    //contains the current correlation matrix
  Eigen::MatrixXd correlationParameters;
  Eigen::MatrixXd prevalenceParameters;
// 
//   //regularization parameter for probabilities
  double alpha;

  pottsMLE(intMat Y_,
           int k_,
           intMat edgeList1_,
           intVec start1_,
           intVec end1_,
           int seed_,
           double alpha_):
  pottsModel(Y_.rows(),
             k_,
             edgeList1_,
             start1_,
             end1_),
             multMixture(Y_,
                         k_,
                         seed_,
                         alpha_)
  {
    matrix corMat;
    corMat.setZero(pottsModel.nCategory,pottsModel.nCategory);
    matrix hMat;
    hMat.setZero(pottsModel.nCategory,1);
    correlationParameters=corMat;
    prevalenceParameters=hMat;
    std::srand(seed_);
  }
  
  
   Rcpp::List EM_SA(int constantIterations,int decreasingIterations,int decreasingStart,double stepSize_,bool stochasticGradient_,bool surrogate_,bool scalarCorrelation_){
//  double EM_SA(int iterations,double stepSize_,bool stochasticGradient_,bool scalarCorrelation_){
    //contains a configuration, drawn approximately via Gibbs sampling,
    //from the marginal distribution of Z
    intVec z;
    z.setZero(pottsModel.n);

    //contains a configuration, drawn approximately via Gibbs sampling,
    //from the conditional distribution of Z given Y
    intVec z_y;
    z_y.setZero(pottsModel.n);

    
    //if the approximate EM surrogate function is used a la Mairal, and Delyon, Lavielle, Moulines
    matrix surrogateQ;
    

    //correlation parameter gradient and value over the run 
    matrix correlationInformation;
    correlationInformation.setZero(decreasingIterations,2);
        
    matrix marginalField;
    matrix conditionalField;
    marginalField.setZero(pottsModel.n,pottsModel.nCategory);
    conditionalField.setZero(pottsModel.n,pottsModel.nCategory);
    matrix onesMatrix;
    onesMatrix.setOnes(pottsModel.n,1);
    
    surrogateQ=multMixture.get_pyz_components(z_y)[0];
// 
//     ////////////////////////////////////////
//     /////do some constant stepsize iterations in order to find a good starting
//     /////point for the decreasing stepsize part of the algorithm
    // for (int iteration=0;iteration<constantIterations;iteration++){
    //   multMixture.computeCondField();
    //   marginalField=onesMatrix*prevalenceParameters.transpose();
    //   conditionalField=marginalField+multMixture.condPotential;
    //   z=pottsModel.gibbs_Sample(z,1,false,marginalField,correlationParameters)[0];
    //   z_y=pottsModel.gibbs_Sample(z_y,1,false,conditionalField,correlationParameters)[0];
    //   updateEtaParameter(z_y,z,scalarCorrelation_,stepSize_,0,0,correlationInformation);
    //   if (stochasticGradient_){
    //     updateMuParameterSG(z_y,stepSize_,0,0,surrogate_);
    //   }
    //   else{
    //     if (surrogate_){
    //       updateMuParameterEM(z_y,surrogateQ,0,0);
    //     }
    //     else{
    //       updateMuParameterEM_shortStep(z_y,0,0);
    //     }
    //   }
    //   if ((iteration%100)==0){
    //     Rcout<<"iteration "<<iteration<<std::endl;
    //     Rcout<<"correlation = "<<correlationParameters(0,0)<<std::endl;
    //   }
    // }
// 
    ////now run the decreasing stepsize part
    for (int iteration=decreasingStart;iteration<(decreasingStart+decreasingIterations);iteration++){
      multMixture.computeCondField();
      marginalField=onesMatrix*prevalenceParameters.transpose();
      conditionalField=marginalField+multMixture.condPotential;
      z=pottsModel.gibbs_Sample(z,1,false,marginalField,correlationParameters)[0];
      z_y=pottsModel.gibbs_Sample(z_y,1,false,conditionalField,correlationParameters)[0];
      updateEtaParameter(z_y,z,scalarCorrelation_,stepSize_,decreasingStart,iteration,correlationInformation);
      if (stochasticGradient_){
        updateMuParameterSG(z_y,stepSize_,decreasingStart,iteration,surrogate_);
      }
      else{
        if (surrogate_){
          updateMuParameterEM(z_y,surrogateQ,decreasingStart,iteration);
        }
        else{
          updateMuParameterEM_shortStep(z_y,decreasingStart,iteration);
        }
      }
      if ((iteration%100)==0){
        Rcout<<"iteration "<<iteration<<std::endl;
        Rcout<<"correlation = "<<correlationParameters(0,0)<<std::endl;
      }
    }
    return Rcpp::List::create(Rcpp::Named("h") = prevalenceParameters,
    Rcpp::Named("correlations") = correlationParameters,
    Rcpp::Named("muParameter") = multMixture.pyz,
    Rcpp::Named("z") = z,
    Rcpp::Named("z_y")=z_y,
    Rcpp::Named("correlationInformation")=correlationInformation);
  }

  void updateEtaParameter(intVec& z_y,intVec& z,bool scalarCorrelation,double stepsize,int decreasingStart,int iteration,matrix& correlationInformation){
    std::vector<intMat> T_y;
    std::vector<intMat> T;
    T_y=pottsModel.computeSufficientStatistics(z_y);
    T=pottsModel.computeSufficientStatistics(z);
    matrix prevDifference=(T_y[0]-T[0]).cast<double>();
    matrix corDifference=(T_y[1]-T[1]).cast<double>();
    
    prevalenceParameters=prevalenceParameters+logisticDensityGradient(prevalenceParameters,stepsize/(iteration+1.0)*(decreasingStart+1.0));
    prevalenceParameters=prevalenceParameters+stepsize*prevDifference/(iteration+1.0)*(decreasingStart+1.0);
    
    if (scalarCorrelation){
      double diagSum=corDifference.diagonal().sum();
      
      correlationInformation(iteration-decreasingStart,0)=correlationParameters(0,0);
      correlationInformation(iteration-decreasingStart,1)=diagSum;
      
      correlationParameters=correlationParameters+logisticDensityGradient(correlationParameters,stepsize/(iteration+1.0)*(decreasingStart+1.0));
      correlationParameters.diagonal().array()+=stepsize*diagSum/(iteration+1.0)*(decreasingStart+1.0);
    }
    else{
      correlationParameters=correlationParameters+logisticDensityGradient(correlationParameters,stepsize/(iteration+1.0)*(decreasingStart+1.0));
      correlationParameters=correlationParameters+stepsize*corDifference/(iteration+1.0)*(decreasingStart+1.0);
    }
  }
  
  //when EM and the surrogate Q function is used
  void updateMuParameterEM(intVec& z_y,matrix& surrogateQ,int decreasingStart,int iteration){
    std::vector<matrix> noiseParameterGradientList_Y=multMixture.get_pyz_components(z_y);
    matrix condProbs_observed=noiseParameterGradientList_Y[0];
    surrogateQ=surrogateQ+(condProbs_observed-surrogateQ)*(decreasingStart+1.0)/(iteration+1.0);
    vec colSums=surrogateQ.colwise().sum();
    multMixture.pyz=surrogateQ*(colSums.asDiagonal().inverse());
  }
  
  //when the EM-like short step version is used
  void updateMuParameterEM_shortStep(intVec& z_y,int decreasingStart,int iteration){
    std::vector<matrix> noiseParameterGradientList_Y=multMixture.get_pyz_components(z_y);
    matrix condProbs_observed=noiseParameterGradientList_Y[0];
    matrix condProbs_expected=noiseParameterGradientList_Y[1];
    double nTrees=multMixture.treeCount+multMixture.pyz.rows()*(alpha-1);
    for (int i=0;i<pottsModel.nCategory;i++){
      double step=(decreasingStart+1.0)/(iteration+1.0)*(1.0/nTrees);
      multMixture.pyz.col(i)=multMixture.pyz.col(i)+step*(condProbs_observed.col(i)-condProbs_expected.col(i));
    }
  }
  
  //when stochastic gradient is used
  void updateMuParameterSG(intVec& z_y,double stepsize,int decreasingStart,int iteration,bool rescale){
    std::vector<matrix> noiseParameterGradientList_Y=multMixture.get_pyz_components(z_y);
    matrix condProbs_observed=noiseParameterGradientList_Y[0];
    matrix condProbs_expected=noiseParameterGradientList_Y[1];
    matrix logpyz;
    logpyz=multMixture.pyz.array().log();
    
    //rescaling the sg step
    if (rescale){
      double nTrees=multMixture.treeCount+multMixture.pyz.rows()*(alpha-1);
      stepsize=1.0/nTrees;
    }
    
    logpyz=logpyz+stepsize*(condProbs_observed-condProbs_expected)*1.0/(iteration+1.0)*(decreasingStart+1.0);
    multMixture.pyz=logpyz.array().exp();
    vec colSums=multMixture.pyz.colwise().sum();
    multMixture.pyz=multMixture.pyz*(colSums.asDiagonal().inverse());
  }
  
  double logSumExp(vec& logProbs){
    double max=logProbs.array().maxCoeff();
    double sumExp=0;
    for (int i=0;i<logProbs.size();i++){
      sumExp+=std::exp(logProbs(i)-max);
    }
    return(std::log(sumExp)+max);
  }

  Rcpp::List crossValidate(intMat& YTest, int nIter,int nBurn){
    matrix pathInformation;
    pathInformation.setZero(nIter,3);
    
    
    bool isCorrelated=correlationParameters.array().abs().maxCoeff()>0.00001;
    matrix frequencies;
    frequencies.setZero(pottsModel.n,pottsModel.nCategory);
    
    matrix marginalField;
    matrix conditionalField;
    marginalField.setZero(pottsModel.n,pottsModel.nCategory);
    conditionalField.setZero(pottsModel.n,pottsModel.nCategory);
    matrix conditionalField2;
    conditionalField2=conditionalField;
    matrix onesMatrix;
    onesMatrix.setOnes(pottsModel.n,1);

    std::vector<intMat> T_y;
    std::vector<intMat> T;
    
    MultinomialMixture multMixture2(YTest,pottsModel.nCategory,multMixture.pyz.rows(),2);
    multMixture2.pyz=multMixture.pyz;

    
    multMixture.computeCondField();
    multMixture2.computeCondField();
    
    marginalField=onesMatrix*prevalenceParameters.transpose();
    conditionalField=marginalField+multMixture.condPotential;
    conditionalField2=marginalField+multMixture2.condPotential;
    
    //will hold the cross-validation log probabilities
    //vec logProbsConditional;
    vec logProbsMarginal;
    //logProbsConditional.setZero(nIter);
    logProbsMarginal.setZero(nIter);

    //compute conditional probabilities for
    //initialization and Gibbs sampling
    intVec z_y_PathIntegral;
    intVec z_y;
    intVec z;
    intVec z_PathIntegral;
    z_y_PathIntegral.setZero(pottsModel.n);
    z_y.setZero(pottsModel.n);
    z.setZero(pottsModel.n);
    z_PathIntegral.setZero(pottsModel.n);
    
  
    if (isCorrelated){//do burnin
      for (int iteration=0; iteration<nBurn;iteration++){
        //the conditional potential is Xbeta+logp(z|y)
        z_y=pottsModel.gibbs_Sample(z_y,1,false,conditionalField,correlationParameters)[0];
  
        if ((iteration % 100)==0){
          Rcout<<"Burnin iteration "<<iteration<<std::endl;
        }
      }
    }
    

    // 
    
    //compute conditional predictive likelihood via gibbs sampling
    //marginal predictive likelihood via path integration
    matrix tempCor=correlationParameters*0;
    matrix zeroCor=correlationParameters*0;
    
    
    double pathSum=0;
    //double logConditionalAverage=0;
    if (isCorrelated){
      for (int iteration=0;iteration<nIter;iteration++){
        z_y=pottsModel.gibbs_Sample(z_y,1,false,conditionalField,correlationParameters)[0];
        
        for (int i=0;i<pottsModel.n;i++){
          frequencies(i,z_y(i))+=(1.0/nIter);
        }
        
        
        z_PathIntegral=pottsModel.gibbs_Sample(z_PathIntegral,1,false,marginalField,tempCor)[0];
        tempCor=tempCor+(1.0/nIter)*correlationParameters;
        z_y_PathIntegral=pottsModel.gibbs_Sample(z_y_PathIntegral,1,false,conditionalField2,tempCor)[0];
        T_y=pottsModel.computeSufficientStatistics(z_y_PathIntegral);
        T=pottsModel.computeSufficientStatistics(z_PathIntegral);
        
        pathInformation(iteration,0)=tempCor(0,0);
        pathInformation(iteration,1)=T_y[1].cast<double>().diagonal().array().sum();
        pathInformation(iteration,2)=T[1].cast<double>().diagonal().array().sum();
        
        double difference=((T_y[1]-T[1]).cast<double>().diagonal().array().sum());
        
        pathSum+=(1.0/nIter)*correlationParameters(0,0)*difference;
        //logProbsConditional(iteration)=computePY_Z(YTest,z_y);
        
        if ((iteration % 100)==0){
          Rcout<<"Iteration "<<iteration<<std::endl;
        }
      }
      //logConditionalAverage=logSumExp(logProbsConditional)-log(nIter);
      // Rcout<<logConditionalAverage<<std::endl;
      // Rcout<<conditionalIndependenceLoglikelihood(YTest);
    }
    else{
    //   logConditionalAverage=conditionalIndependenceLoglikelihood(YTest);
      multMixture.computeCondProbs();
      frequencies=multMixture.condProbs;
    }
    
    double logMarginalAverage=marginalIndependenceLoglikelihood(YTest)+pathSum;

    return Rcpp::List::create(Rcpp::Named("marginalCVLogLike")=logMarginalAverage,
                              Rcpp::Named("pathInformation")=pathInformation,
                              Rcpp::Named("pathIntegral")=pathSum,
                              Rcpp::Named("classificationFrequencies")=frequencies);
  }

  double computePY_Z(intMat& YTest_, intVec& Z_YTrain){
    double val=0;

    for (int i=0;i<multMixture.n;i++){
      val=val+computeLogConditionalProbability_i(i,YTest_,Z_YTrain);
    }
    return(val);
  }

  double marginalIndependenceLoglikelihood(intMat& Y){
    double independenceLogProbability=0;
    matrix classProbs=prevalenceParameters.array().exp();
    classProbs=classProbs.array()/(classProbs.array().sum());
    for (int i=0;i<pottsModel.n;i++){
      double p_i=0;
      for (int j=0;j<pottsModel.nCategory;j++){
        double temp=classProbs(j,0);
        for (int m=0;m<multMixture.pyz.rows();m++){
          temp=temp*std::pow(multMixture.pyz(m,j),Y(i,m));
        }
        p_i+=temp;
      }
      independenceLogProbability+=std::log(p_i);
    }
    return(independenceLogProbability);
  }
  
  double conditionalIndependenceLoglikelihood(intMat& Y){
    double independenceLogProbability=0;

    for (int i=0;i<pottsModel.n;i++){
      vec classProbs=prevalenceParameters.col(0);
      for (int j=0;j<pottsModel.nCategory;j++){
        classProbs(j)+=multMixture.condPotential(i,j);
      }
      classProbs=classProbs.array().exp();
      classProbs=classProbs.array()/(classProbs.array().sum());
      double p_i=0;
      for (int j=0;j<pottsModel.nCategory;j++){
        double temp=classProbs(j);
        for (int m=0;m<multMixture.nSpecies;m++){
          temp=temp*std::pow(multMixture.pyz(m,j),Y(i,m));
        }
        p_i+=temp;
      }
      independenceLogProbability+=std::log(p_i);
    }
    return(independenceLogProbability);
  }
  
  double computeLogConditionalProbability_i(int i, intMat& YTest_,intVec& z_YTrain){
    double val=0;
    for (int j=0;j<multMixture.nSpecies;j++){
      val=val+YTest_(i,j)*std::log(multMixture.pyz(j,z_YTrain(i)));
    }
    return(val);
  }

  matrix logisticDensityGradient(matrix parameter,double stepSize){
    double alpha=2;
    matrix step=0*parameter;
    for (int i1=0;i1<parameter.rows();i1++){
      for (int i2=0;i2<parameter.cols();i2++){
        double parameterValue=parameter(i1,i2);
        double temp=-2*(exp(parameterValue)-exp(-parameterValue));
        temp=stepSize*temp/(exp(parameterValue)+exp(-parameterValue));
        step(i1,i2)=temp*(alpha-1);
      }
    }

    return step;
  }

};