% mean coherence of 1000 random matrices
rng(11);
coherences = [];
for i=1:1000
    U = orth(rand(1000, 50));
    coherence = max(sum(U.^2, 2));
    coherences = [coherences, coherence];
end
disp(mean(coherences))

% coherence of a matrix with one nonzero
% element in a specific column
A = rand(1000, 50);
A(:,1) = 0;
A(1,1) = 1;
U = orth(A);
coherence_1 = max(sum(U.^2, 2));
disp(coherence_1)
