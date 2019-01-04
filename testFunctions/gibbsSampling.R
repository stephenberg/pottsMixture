library(igraph)
library(pottsMixture)
library(Matrix)
graph=make_lattice(c(100,100))
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
edge=as.matrix(edge[,2]-1)
start=as.matrix(start-1)
end=as.matrix(end-1)

# double gibbsSample(int k_,
#                    Eigen::MatrixXi edge1_,
#                    Eigen::MatrixXi start1_,
#                    Eigen::MatrixXi end1_,
#                    Eigen::MatrixXi inGraph_,
#                    int seed_,
#                    int nIter_,
#                    bool offDiagonal_,
#                    bool fullPath_,
#                    Eigen::VectorXd h_,
#                    Eigen::MatrixXd noiseProbabilities_,
#                    std::vector<Eigen::MatrixXd> correlations_)

k=8
inGraph=as.matrix(rep(1,10000))
seed=33
nIter=100000
offDiagonal=FALSE
fullPath=FALSE
h=rep(0,k)
correlations=list(1.36*diag(8))

z=gibbsSample(k=k,edge1_ = edge,start1_ = start,end1_ = end,inGraph_ = inGraph,seed_ = seed,nIter_ = nIter,offDiagonal_ = offDiagonal,fullPath_ = fullPath,h_ = h,correlations_ = correlations)
image(matrix(z[[1]],ncol=100))
diag(A)=0
y=factor(z[[1]])
fit=MPLE(X = matrix(rep(1,10000),ncol=1),A=A,y=y)
fit$gammaHat
