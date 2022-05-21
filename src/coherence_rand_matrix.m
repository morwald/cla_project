rand('state', 42);
n_matrices = 1000; n_rows = 1000; n_cols = 50;

% b) average coherences
coherences = [];
for n=1:n_matrices
    A = rand(n_rows, n_cols);
    if ~(rank(A) == n_cols)
        fprintf("rank deficient random matrix with rank: %i\n", rank(A));
    end
    [Q, R] = qr(A, 0);
    coherence = max(sum(Q .^ 2, 2));
    coherences = [coherences, coherence];
end

mean_coherence = mean(coherences);
fprintf("Mean coherence of random matrices: %d\n", mean_coherence);


% b) example matrix with dimension (1000, 50) with coherence = 1
A = rand(n_rows, n_cols);
A(:, 1) = 0;
%A(1, 1) = 0; % coherence=1 as well with one nnz entry
[Q, R] = qr(A, 0);
coherence = max(sum(Q .^ 2, 2));

fprintf("Example matrix with worst possible coherence: %d\n", coherence);

