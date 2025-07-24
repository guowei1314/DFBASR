clear all

load("YaleA_PCA.mat");  %YaleA_PCA.mat is obtained via "multi_view_PCA_data.m".

alpha = 10;
beta = 1;
gamma = 10;
deta = 1;


Z_star = JDFBSR_clustering(X, K, alpha, beta, gamma,deta);

for i = 1:10
    Clus = SpectralClustering(Z_star,size(unique(gt),1));
    result(i,:) = EvaluationMetrics(gt,Clus);
end
result1 = mean(result);
fprintf("ACC:%5.4f,NMI:%5.4f,alpha:%5.4f,beta:%5.4f,gamma:%5.4f,deta:%5.4f\n",result1(1), result1(2), alpha, beta, gamma,deta);
