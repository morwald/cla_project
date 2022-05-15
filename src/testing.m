rand('state', 2378)
randn('state', 23984)
m = 20000; n = 100;
A = rand(m, n); b = rand(m, 1);
[U, S, V] = svd(A, 0);
S = diag(linspace(1, 10^6, 100));
A = U * S * V';

A_bad = A;
A_bad(:, 1) = zeros(size(A, 1), 1);
[Q, R] = qr(A_bad, 0);
coherence = max(sum(Q .^ 2, 2));

