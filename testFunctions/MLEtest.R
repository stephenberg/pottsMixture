library(igraph)
library(pottsMixture)
library(Matrix)
source("C:/Users/Stephen Berg/Dropbox/Research - Berg (1)/SpatialClassification/Manuscript/UtilityFunctions/pottsSimulationFunctions.R")

graph=make_lattice(c(50,50))
A=get.adjacency(graph)
diag(A)=1
graph=graph_from_adjacency_matrix(A)
edge=get.edgelist(graph)

edge=rbind(edge,cbind(edge[,2],edge[,1]))
edge=unique(edge)
edge=edge[order(edge[,1]),]
counts=table(edge[,1])
start=c(1,cumsum(counts[1:(length(counts)-1)])+1)
end=cumsum(counts)
edge=as.matrix(edge-1)
start=(start-1)
end=(end-1)
k=8
seed=33
nIter=5000
load("C:/Users/Stephen Berg/Dropbox/Research - Berg (1)/SpatialClassification/Manuscript/PLS-Data/SpatialLattice/plsLatticeObjects4kShape.RData")
z=pottsMixture::gibbsSample(k,edge,start,end,seed,2000,FALSE,FALSE,matrix(rep(0,k),ncol=1),1.2*diag(k))
table(z[[1]])



probs=matrix(exp(runif(33*k)*15),ncol=k)
probs=probs%*%solve(diag(apply(probs,2,sum)))
probs=(probs+0.01)
probs=probs%*%solve(diag(apply(probs,2,sum)))

#probs=cbind(c(0.02,0.02,0.02,0.02,0.02,0.9),c(0.02,0.02,0.02,0.02,0.9,0.02),c(0.02,0.02,0.02,0.9,0.02,0.02),c(0.02,0.02,0.9,0.02,0.02,0.02),c(0.02,0.9,0.02,0.02,0.02,0.02))
zMat=matrix(0,length(z[[1]]),k)
for (i in 1:length(z[[1]])){
  zMat[i,z[[1]][i]+1]=1
}
Y=generateYPotts(zMat,probs,6)
Yt=generateTestTrain(Y,n=2500)

fit=EMSA(Yt[[1]],k,edge,start,end,seed,0,5000,1000,TRUE,0.05/(100^2),2,FALSE,FALSE)
fitSG=EMSA(Yt[[1]],k,edge,start,end,seed,0,5000,1000,TRUE,0.01/(100^2),2,TRUE,FALSE)
#fitSurrogate=EMSA(Yt[[1]],3,edge,start,end,seed,0,4000,1000,TRUE,0.1/(100^2),2,FALSE,TRUE)
fit2=pottsMixture::estimateMixture(Yt[[1]],k_ = 8,nIter_ = 1000,alpha_ = 2,seed=33)

#fit3=kernlab::specc(Y[1:5000,],10)

cvResults=cv_posteriorPredictive(Yt[[1]],Yt[[2]],edge,start,end,seed,10000,1000,fit$h,fit$muParameter,fit$correlations)
cvResults2=cv_posteriorPredictive(Yt[[1]],Yt[[2]],edge,start,end,seed,200,100,matrix(fit2$h,ncol=1),fit2$pyz,diag(3)*0)
# 
# cvResults_1=cv_posteriorPredictive(Yt[[1]],Yt[[2]],edge,start,end,seed,1000,1000,fit$h,fit$muParameter,fit$correlations)
# cvResults2_1=cv_posteriorPredictive(Yt[[1]],Yt[[2]],edge,start,end,seed,1000,1000,matrix(fit2$h,ncol=1),fit2$pyz,diag(3)*0)
# # cvResults_truth=cv_posteriorPredictive(Yt[[1]],Yt[[2]],edge,start,end,seed,20000,2000,fit$h*0,probs,1.2*diag(8))
# # cvResults3=cv_posteriorPredictive(Yt[[1]],Yt[[2]],edge,start,end,seed,20000,2000,fit$h,fit$muParameter,1.32*diag(8))
# # cvResults4=cv_posteriorPredictive(Yt[[1]],Yt[[2]],edge,start,end,seed,20000,2000,fit$h,fit$muParameter,1.34*diag(8))
# # cvResults5=cv_posteriorPredictive(Yt[[1]],Yt[[2]],edge,start,end,seed,20000,2000,fit$h,fit$muParameter,1.36*diag(8))
# cvResults$conditionalCVLogLike
# cvResults$marginalCVLogLike
# cvResults2$conditionalCVLogLike
# cvResults2$marginalCVLogLike
# # cvResults_truth
# 
# 
# 
# 
# ###########################
# #pam comparison
# fitPam=cluster::pam(Yt[[1]],k=8)
# 
# muPam=matrix(0,dim(probs)[1],dim(probs)[2])
# for (i in 1:dim(probs)[2]){
#   muPam[,i]=apply(Yt[[1]][fitPam$clustering==i,],2,sum)
#   muPam[,i]=muPam[,i]/sum(muPam[,i])
# }
# 
# emError=0
# indEMError=0
# pamError=0
# 
# for (i in 1:dim(Yt[[2]])[1]){
#   emClass=which.max(cvResults$classificationFrequencies[i,])
#   yMean=Yt[[2]][i,]/sum(Yt[[2]][i,])
#   emError=emError+sum((yMean-fit$muParameter[,emClass])^2)
#   
#   indEMClass=which.max(cvResults2$classificationFrequencies[i,])
#   indEMError=indEMError+sum((yMean-fit2$pyz[,indEMClass])^2)
#   
#   
#   pamClass=fitPam$clustering[i]
#   pamError=pamError+sum((yMean-muPam[,pamClass])^2)
# }
# 
# emError
# indEMError
# pamError
# 
# 
# settings=NULL
# for (i in c(8,12,16,24)){
#   for (j in c(1,2)){
#     for (k in c(4,2,1)){
#       for (r in 1:10){
#         settings=rbind(settings,c(i,j,k))
#       }
#     }
#   }
# }
# 
# 
