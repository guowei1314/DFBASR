function W = solve_W(X, A, Z, Q, lambda)


% 求解优化问题：min_W ||X - A*W*Z||_F^2 + lambda * w'*Q*w
% 约束：sum(w) = N, w >= 0
% 输入：
%   X: d x N 矩阵
%   A: d x N 矩阵
%   Z: N x N 矩阵
%   Q: N x N 矩阵（对称）
%   lambda: 正则化参数
% 输出：
%   W: N x N 对角矩阵，对角线为最优的 w

% 获取维度
N = size(A, 2);

% 计算 c = diag(Z * X' * A)
c = diag(Z * X' * A);

% 计算 H = (A' * A) .* (Z * Z')'
H = (A' * A) .* (Z * Z')';

% 构造 M = H + lambda * Q
M = H + lambda * Q;

% 转换为二次规划标准形式：
% min (1/2) w' * P * w + q' * w
% s.t. Aeq * w = beq, lb <= w
P = 2 * M;  % 因为目标函数是 w'*M*w - 2*c'*w
q = -2 * c;

% 约束：sum(w) = N, N >= 0
Aeq = ones(1, N);
beq = N;
lb = zeros(N, 1);
ub = [];  % 无上界

% 使用 quadprog 求解
options = optimoptions('quadprog', 'Display', 'off');
w = quadprog(P, q, [], [], Aeq, beq, lb, ub, [], options);

% 构建对角矩阵 W
W = diag(w);
end