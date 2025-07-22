clear all
clc

load("ORL_mtv.mat");
[K] = multi_view_pca_unified_k(X,size(unique(gt),1));
save_name = strcat("ORL_PCA.mat");
save(save_name,"X","gt","K");


