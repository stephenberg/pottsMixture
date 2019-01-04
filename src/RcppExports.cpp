// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// estimateMixture
Rcpp::List estimateMixture(Eigen::MatrixXi Y_, int k_, int nIter_, int seed_, double alpha_);
RcppExport SEXP _pottsMixture_estimateMixture(SEXP Y_SEXP, SEXP k_SEXP, SEXP nIter_SEXP, SEXP seed_SEXP, SEXP alpha_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXi >::type Y_(Y_SEXP);
    Rcpp::traits::input_parameter< int >::type k_(k_SEXP);
    Rcpp::traits::input_parameter< int >::type nIter_(nIter_SEXP);
    Rcpp::traits::input_parameter< int >::type seed_(seed_SEXP);
    Rcpp::traits::input_parameter< double >::type alpha_(alpha_SEXP);
    rcpp_result_gen = Rcpp::wrap(estimateMixture(Y_, k_, nIter_, seed_, alpha_));
    return rcpp_result_gen;
END_RCPP
}
// estimateMixtureWithErrors
Rcpp::List estimateMixtureWithErrors(Eigen::MatrixXi Y_, Eigen::MatrixXi Y_test, int k_, int nIter_, int seed_, double alpha_);
RcppExport SEXP _pottsMixture_estimateMixtureWithErrors(SEXP Y_SEXP, SEXP Y_testSEXP, SEXP k_SEXP, SEXP nIter_SEXP, SEXP seed_SEXP, SEXP alpha_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXi >::type Y_(Y_SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXi >::type Y_test(Y_testSEXP);
    Rcpp::traits::input_parameter< int >::type k_(k_SEXP);
    Rcpp::traits::input_parameter< int >::type nIter_(nIter_SEXP);
    Rcpp::traits::input_parameter< int >::type seed_(seed_SEXP);
    Rcpp::traits::input_parameter< double >::type alpha_(alpha_SEXP);
    rcpp_result_gen = Rcpp::wrap(estimateMixtureWithErrors(Y_, Y_test, k_, nIter_, seed_, alpha_));
    return rcpp_result_gen;
END_RCPP
}
// estimateInitializedMixture
Rcpp::List estimateInitializedMixture(Eigen::MatrixXi Y_, int k_, int nIter_, int seed_, double alpha_, Eigen::MatrixXd initialPYZ_, Eigen::VectorXd initialH_);
RcppExport SEXP _pottsMixture_estimateInitializedMixture(SEXP Y_SEXP, SEXP k_SEXP, SEXP nIter_SEXP, SEXP seed_SEXP, SEXP alpha_SEXP, SEXP initialPYZ_SEXP, SEXP initialH_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXi >::type Y_(Y_SEXP);
    Rcpp::traits::input_parameter< int >::type k_(k_SEXP);
    Rcpp::traits::input_parameter< int >::type nIter_(nIter_SEXP);
    Rcpp::traits::input_parameter< int >::type seed_(seed_SEXP);
    Rcpp::traits::input_parameter< double >::type alpha_(alpha_SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type initialPYZ_(initialPYZ_SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type initialH_(initialH_SEXP);
    rcpp_result_gen = Rcpp::wrap(estimateInitializedMixture(Y_, k_, nIter_, seed_, alpha_, initialPYZ_, initialH_));
    return rcpp_result_gen;
END_RCPP
}
// estimateInitializedMixturewithErrors
Rcpp::List estimateInitializedMixturewithErrors(Eigen::MatrixXi Y_, Eigen::MatrixXi Y_test, int k_, int nIter_, int seed_, double alpha_, Eigen::MatrixXd initialPYZ_, Eigen::VectorXd initialH_);
RcppExport SEXP _pottsMixture_estimateInitializedMixturewithErrors(SEXP Y_SEXP, SEXP Y_testSEXP, SEXP k_SEXP, SEXP nIter_SEXP, SEXP seed_SEXP, SEXP alpha_SEXP, SEXP initialPYZ_SEXP, SEXP initialH_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXi >::type Y_(Y_SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXi >::type Y_test(Y_testSEXP);
    Rcpp::traits::input_parameter< int >::type k_(k_SEXP);
    Rcpp::traits::input_parameter< int >::type nIter_(nIter_SEXP);
    Rcpp::traits::input_parameter< int >::type seed_(seed_SEXP);
    Rcpp::traits::input_parameter< double >::type alpha_(alpha_SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type initialPYZ_(initialPYZ_SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type initialH_(initialH_SEXP);
    rcpp_result_gen = Rcpp::wrap(estimateInitializedMixturewithErrors(Y_, Y_test, k_, nIter_, seed_, alpha_, initialPYZ_, initialH_));
    return rcpp_result_gen;
END_RCPP
}
// gibbsSample
std::vector<Eigen::VectorXi> gibbsSample(int k_, Eigen::MatrixXi edge1_, Eigen::VectorXi start1_, Eigen::VectorXi end1_, int seed_, int nIter_, bool offDiagonal_, bool fullPath_, Eigen::MatrixXd h_, Eigen::MatrixXd correlations_);
RcppExport SEXP _pottsMixture_gibbsSample(SEXP k_SEXP, SEXP edge1_SEXP, SEXP start1_SEXP, SEXP end1_SEXP, SEXP seed_SEXP, SEXP nIter_SEXP, SEXP offDiagonal_SEXP, SEXP fullPath_SEXP, SEXP h_SEXP, SEXP correlations_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type k_(k_SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXi >::type edge1_(edge1_SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type start1_(start1_SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type end1_(end1_SEXP);
    Rcpp::traits::input_parameter< int >::type seed_(seed_SEXP);
    Rcpp::traits::input_parameter< int >::type nIter_(nIter_SEXP);
    Rcpp::traits::input_parameter< bool >::type offDiagonal_(offDiagonal_SEXP);
    Rcpp::traits::input_parameter< bool >::type fullPath_(fullPath_SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type h_(h_SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type correlations_(correlations_SEXP);
    rcpp_result_gen = Rcpp::wrap(gibbsSample(k_, edge1_, start1_, end1_, seed_, nIter_, offDiagonal_, fullPath_, h_, correlations_));
    return rcpp_result_gen;
END_RCPP
}
// EMSA
Rcpp::List EMSA(Eigen::MatrixXi Y_, int k_, Eigen::MatrixXi edge1_, Eigen::VectorXi start1_, Eigen::VectorXi end1_, int seed_, int constantIterations_, int decreasingIterations_, int decreasingStart_, bool scalarCorrelations_, double stepSize_, double alpha_, bool stochasticGradient_, bool surrogate_);
RcppExport SEXP _pottsMixture_EMSA(SEXP Y_SEXP, SEXP k_SEXP, SEXP edge1_SEXP, SEXP start1_SEXP, SEXP end1_SEXP, SEXP seed_SEXP, SEXP constantIterations_SEXP, SEXP decreasingIterations_SEXP, SEXP decreasingStart_SEXP, SEXP scalarCorrelations_SEXP, SEXP stepSize_SEXP, SEXP alpha_SEXP, SEXP stochasticGradient_SEXP, SEXP surrogate_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXi >::type Y_(Y_SEXP);
    Rcpp::traits::input_parameter< int >::type k_(k_SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXi >::type edge1_(edge1_SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type start1_(start1_SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type end1_(end1_SEXP);
    Rcpp::traits::input_parameter< int >::type seed_(seed_SEXP);
    Rcpp::traits::input_parameter< int >::type constantIterations_(constantIterations_SEXP);
    Rcpp::traits::input_parameter< int >::type decreasingIterations_(decreasingIterations_SEXP);
    Rcpp::traits::input_parameter< int >::type decreasingStart_(decreasingStart_SEXP);
    Rcpp::traits::input_parameter< bool >::type scalarCorrelations_(scalarCorrelations_SEXP);
    Rcpp::traits::input_parameter< double >::type stepSize_(stepSize_SEXP);
    Rcpp::traits::input_parameter< double >::type alpha_(alpha_SEXP);
    Rcpp::traits::input_parameter< bool >::type stochasticGradient_(stochasticGradient_SEXP);
    Rcpp::traits::input_parameter< bool >::type surrogate_(surrogate_SEXP);
    rcpp_result_gen = Rcpp::wrap(EMSA(Y_, k_, edge1_, start1_, end1_, seed_, constantIterations_, decreasingIterations_, decreasingStart_, scalarCorrelations_, stepSize_, alpha_, stochasticGradient_, surrogate_));
    return rcpp_result_gen;
END_RCPP
}
// initializedEMSA
Rcpp::List initializedEMSA(Eigen::MatrixXi Y_, int k_, Eigen::MatrixXi edge1_, Eigen::VectorXi start1_, Eigen::VectorXi end1_, int seed_, int constantIterations_, int decreasingIterations_, int decreasingStart_, bool scalarCorrelations_, double stepSize_, double alpha_, bool stochasticGradient_, bool surrogate_, Eigen::MatrixXd initialH_, Eigen::MatrixXd initialCorrelations_, Eigen::MatrixXd initialMu_);
RcppExport SEXP _pottsMixture_initializedEMSA(SEXP Y_SEXP, SEXP k_SEXP, SEXP edge1_SEXP, SEXP start1_SEXP, SEXP end1_SEXP, SEXP seed_SEXP, SEXP constantIterations_SEXP, SEXP decreasingIterations_SEXP, SEXP decreasingStart_SEXP, SEXP scalarCorrelations_SEXP, SEXP stepSize_SEXP, SEXP alpha_SEXP, SEXP stochasticGradient_SEXP, SEXP surrogate_SEXP, SEXP initialH_SEXP, SEXP initialCorrelations_SEXP, SEXP initialMu_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXi >::type Y_(Y_SEXP);
    Rcpp::traits::input_parameter< int >::type k_(k_SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXi >::type edge1_(edge1_SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type start1_(start1_SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type end1_(end1_SEXP);
    Rcpp::traits::input_parameter< int >::type seed_(seed_SEXP);
    Rcpp::traits::input_parameter< int >::type constantIterations_(constantIterations_SEXP);
    Rcpp::traits::input_parameter< int >::type decreasingIterations_(decreasingIterations_SEXP);
    Rcpp::traits::input_parameter< int >::type decreasingStart_(decreasingStart_SEXP);
    Rcpp::traits::input_parameter< bool >::type scalarCorrelations_(scalarCorrelations_SEXP);
    Rcpp::traits::input_parameter< double >::type stepSize_(stepSize_SEXP);
    Rcpp::traits::input_parameter< double >::type alpha_(alpha_SEXP);
    Rcpp::traits::input_parameter< bool >::type stochasticGradient_(stochasticGradient_SEXP);
    Rcpp::traits::input_parameter< bool >::type surrogate_(surrogate_SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type initialH_(initialH_SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type initialCorrelations_(initialCorrelations_SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type initialMu_(initialMu_SEXP);
    rcpp_result_gen = Rcpp::wrap(initializedEMSA(Y_, k_, edge1_, start1_, end1_, seed_, constantIterations_, decreasingIterations_, decreasingStart_, scalarCorrelations_, stepSize_, alpha_, stochasticGradient_, surrogate_, initialH_, initialCorrelations_, initialMu_));
    return rcpp_result_gen;
END_RCPP
}
// cv_posteriorPredictive
Rcpp::List cv_posteriorPredictive(Eigen::MatrixXi YTrain_, Eigen::MatrixXi YTest_, Eigen::MatrixXi edge1_, Eigen::VectorXi start1_, Eigen::VectorXi end1_, int seed_, int nIter_, int nBurn_, Eigen::MatrixXd H_, Eigen::MatrixXd muParameter_, Eigen::MatrixXd correlations_);
RcppExport SEXP _pottsMixture_cv_posteriorPredictive(SEXP YTrain_SEXP, SEXP YTest_SEXP, SEXP edge1_SEXP, SEXP start1_SEXP, SEXP end1_SEXP, SEXP seed_SEXP, SEXP nIter_SEXP, SEXP nBurn_SEXP, SEXP H_SEXP, SEXP muParameter_SEXP, SEXP correlations_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXi >::type YTrain_(YTrain_SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXi >::type YTest_(YTest_SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXi >::type edge1_(edge1_SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type start1_(start1_SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type end1_(end1_SEXP);
    Rcpp::traits::input_parameter< int >::type seed_(seed_SEXP);
    Rcpp::traits::input_parameter< int >::type nIter_(nIter_SEXP);
    Rcpp::traits::input_parameter< int >::type nBurn_(nBurn_SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type H_(H_SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type muParameter_(muParameter_SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type correlations_(correlations_SEXP);
    rcpp_result_gen = Rcpp::wrap(cv_posteriorPredictive(YTrain_, YTest_, edge1_, start1_, end1_, seed_, nIter_, nBurn_, H_, muParameter_, correlations_));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_pottsMixture_estimateMixture", (DL_FUNC) &_pottsMixture_estimateMixture, 5},
    {"_pottsMixture_estimateMixtureWithErrors", (DL_FUNC) &_pottsMixture_estimateMixtureWithErrors, 6},
    {"_pottsMixture_estimateInitializedMixture", (DL_FUNC) &_pottsMixture_estimateInitializedMixture, 7},
    {"_pottsMixture_estimateInitializedMixturewithErrors", (DL_FUNC) &_pottsMixture_estimateInitializedMixturewithErrors, 8},
    {"_pottsMixture_gibbsSample", (DL_FUNC) &_pottsMixture_gibbsSample, 10},
    {"_pottsMixture_EMSA", (DL_FUNC) &_pottsMixture_EMSA, 14},
    {"_pottsMixture_initializedEMSA", (DL_FUNC) &_pottsMixture_initializedEMSA, 17},
    {"_pottsMixture_cv_posteriorPredictive", (DL_FUNC) &_pottsMixture_cv_posteriorPredictive, 11},
    {NULL, NULL, 0}
};

RcppExport void R_init_pottsMixture(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}