% rng(11);
% U = orth(rand(20000,400));
% S = diag(linspace(1,1e5,400));
% V = orth(rand(400));
% A = U*S*V';
% b = rand(size(A, 1), 1);

rng(11);
A = [ diag(linspace(1,1e5,400)); zeros(19600,400) ];
A = A + 1e-8*ones(20000,400);
b = rand(size(A, 1), 1);

x = A \ b;

errors = [];
gammas = linspace(2, 10, 10);
for gamma = gammas
    fprintf("gamma: %d\n", gamma);
    x_tilde = blendenpik(A, b, gamma, "DCT");
    errors = [errors, norm(x - x_tilde)];
end
