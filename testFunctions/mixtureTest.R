rm(list=ls())
library(pottsMixture)
load("C:/Users/Stephen Berg/Dropbox/Research - Berg (1)/SpatialClassification/Manuscript/Results/shortStep/SAEM2k_8cat_cv1_rep2.RData")


nIter=200

fitMix=estimateMixtureWithErrors(YTrain,YTest,8,100,33,2)

YTrain2=YTrain[apply(YTrain,1,sum)>0,]
for (i in 1:dim(YTrain2)[1]){
  YTrain2[i,]=YTrain2[i,]/sum(YTrain2[i,])
}

fitKmeans=kmeans(YTrain2,8)
h=rep(0,8)
for (i in 1:8){
  h[i]=mean(fitKmeans$cluster==i)
}
h=log(h)
pyzKMeans=t(fitKmeans$centers)
fitMixKMeans=estimateInitializedMixturewithErrors(YTrain[apply(YTrain,1,sum)>0,],YTest[apply(YTrain,1,sum)>0,],8,100,0,2,pyzKMeans,h)





fitKmeans2=kmeans(YTrain[apply(YTrain,1,sum)>0,],8)
h=rep(0,8)
for (i in 1:8){
  h[i]=mean(fitKmeans2$cluster==i)
}
h=log(h)
pyzKMeans=t(fitKmeans2$centers)
pyzKMeans=pyzKMeans%*%solve(diag(apply(pyzKMeans,2,sum)))
fitMixKMeans2=estimateInitializedMixturewithErrors(YTrain[apply(YTrain,1,sum)>0,],YTest[apply(YTrain,1,sum)>0,],8,100,0,2,pyzKMeans,h)


p=runif(8)
h=log(p/sum(p))
pyz=matrix(exp(runif(8*33)*1),ncol=8)
pyz=pyz%*%solve(diag(apply(pyz,2,sum)))

#fitMixRandom0=estimateInitializedMixturewithErrors(YTrain,8,500,33,2,pyz,h)
fitMixRandom=estimateMixtureWithErrors(YTrain[apply(YTrain,1,sum)>0,],YTest[apply(YTrain,1,sum)>0,],8,100,27,2)
errorsRandom=fitMixRandom$errors
errors=fitMix$errors
range1=max(errors[1,])-min(errors[1,])
range2=max(errors[2,])-min(errors[2,])

errors[2,]
# mu=fitMix$pyzStart
# h=fitMix$hStart
# p=exp(h)/sum(exp(h))
# 
# 
# for (iter in 1:nIter){
# condProbs=matrix(0,dim(YTrain)[1],8)
# 
# for (i in 1:dim(YTrain)[1]){
#   for (j in 1:8){
#     condProbs[i,j]=p[j]
#     for (m in 1:33){
#       condProbs[i,j]=condProbs[i,j]*mu[m,j]^YTrain[i,m]
#     }
#   }
#   condProbs[i,]=condProbs[i,]/sum(condProbs[i,])
# }
# 
# p=apply(condProbs,2,sum)+1
# p=p/sum(p)
# 
# p
# exp(fitMix$h)
# 
# mu=matrix(0,33,8)
# for (j in 1:8){
#   for (i in 1:dim(YTrain)[1]){
#     mu[,j]=mu[,j]+condProbs[i,j]*YTrain[i,]
#   }
# }
# mu=mu+1
# mu=mu%*%solve(diag(apply(mu,2,sum)))
# }
# 
# 
# max(abs(exp(fitMix$h)-p))
# max(abs(fitMix$pyz-mu))