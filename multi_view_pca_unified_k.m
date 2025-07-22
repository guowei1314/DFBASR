function [k_values] = multi_view_pca_unified_k(data, num_clusters)
% MULTI_VIEW_PCA_UNIFIED_K 多视图数据PCA降维处理
% 输入:
%   data - 1×V的cell数组，每个cell包含d_v×N矩阵(第v个视图的数据)
%   num_clusters - 整数，指定的簇数目
% 输出:
%   reduced_data - 1×V的cell数组，包含降维后的数据(k×N矩阵)
%   best_k - 计算得到的最佳k值

V = length(data); % 视图数量
k_values = zeros(1, V); % 存储每个视图的k值
reduced_data = cell(1, V); % 存储降维后的数据

% 对每个视图进行处理
for v = 1:V
    X = data{v};
    [d_v, N] = size(X);
    
    % 中心化数据
    X_centered = X - mean(X, 2);
    
    % 计算协方差矩阵
    covariance = (X_centered * X_centered') / (N - 1);
    
    % PCA分解
    [eigenvectors, eigenvalues] = eig(covariance);
    eigenvalues = diag(eigenvalues);
    [eigenvalues, idx] = sort(eigenvalues, 'descend');
    eigenvectors = eigenvectors(:, idx);
    
    % 确定保留的主成分数量(解释方差>95%)
    total_variance = sum(eigenvalues);
    explained_variance = cumsum(eigenvalues) / total_variance;
    k = find(explained_variance >= 0.95, 1, 'first');
    
    % 如果没有达到95%，保留所有维度
    if isempty(k)
        k = d_v;
    end
    
    k_values(v) = k;
    if k_values(v) < num_clusters
       k_values(v) = num_clusters;
        if k_values(v) > d_v
            k_values(v) = d_v;
        end
    end
    
    
    % 降维处理
    W = eigenvectors(:, 1:k); % 投影矩阵
   
    reduced_data{v} = W' * X_centered; % 降维后的数据
end
end