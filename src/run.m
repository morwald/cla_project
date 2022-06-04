%% Defining Matrices
clear all; close all; clc;
tol = 1e-10; maxit = 200; verbose=false;

rng(11);
U = orth(rand(20000, 400));
S = diag(linspace(1, 1e5, 400));
V = orth(rand(400));
A1 = U*S*V';
b1 = rand(size(A1, 1), 1);
x1 = A1 \ b1;

A2 = [ diag(linspace(1,1e5,400)); zeros(19600,400) ];
A2 = A2 + 1e-8*ones(20000,400);
b2 = rand(size(A2, 1), 1);
x2 = A2 \ b2;

%% part e) Convergence of (in)coherent matrices for different gammas
gammas = 2:1:16;

% i) Incoherent, ill-conditioned matrix 
fprintf("Incoherent, ill-conditioned matrix\n");
fprintf("cond(A1): %d\n", cond(A1));
fprintf("cond(A1'*A1): %d\n", cond(A1'*A1));

errorsA1 = [];
itersA1 = [];
for gamma = gammas
    fprintf("gamma: %d\n", gamma);
    [x_tilde, iters, resvecA1] = blendenpik(A1, b1, gamma, "minres", ...
                                            "DCT", tol, maxit, verbose);
    
    errorsA1 = [errorsA1, norm(x1 - x_tilde)];
    itersA1 = [itersA1, iters];
end
fprintf("\n\n");

% ii) Coherent, ill-conditioned matrix
fprintf("Coherent, ill-conditioned matrix\n");
fprintf("cond(A2): %d \n", cond(A2));
fprintf("cond(A2'*A2): %d \n", cond(A2'*A2));

errorsA2 = [];
itersA2 = [];
for gamma = gammas
    fprintf("gamma: %d\n", gamma);
    [x_tilde, iters, resvecA2] = blendenpik(A2, b2, gamma, "minres", ...
                                            "DCT", tol, maxit, verbose);
    
    errorsA2 = [errorsA2, norm(x2 - x_tilde)];
    itersA2 = [itersA2, iters];
end
fprintf("\n\n");

figure(1);
plot(gammas, itersA1, '-*');
hold on;
plot(gammas, itersA2, '-o');
title("A1 and A2 convergence vs \gamma (tol=" + tol + ")");
legend("A1 MinRes Iterations in Blendenpik", "A2 MinRes Iterations in Blendenpik");
xlabel("\gamma");
ylabel("Number of iterations to convergence");
ylim([0, maxit * 1.1]);
grid on;
hold off;

%% part f) Convergence of inner LSQR/ MINRES steps in Blendenpik
gamma = 5; maxit=200;

[x1m, iters, resvecA1_mres] = blendenpik(A1, b1, gamma, "minres", "DCT", ... 
                                        tol, maxit, verbose);
[x1l, iters, resvecA1_lsqr] = blendenpik(A1, b1, gamma, "lsqr", "DCT", ... 
                                        tol, maxit, verbose);

[x2m, iters, resvecA2_mres] = blendenpik(A2, b2, gamma, "minres", "DCT", ... 
                                        tol, maxit, verbose);
[x2l, iters, resvecA2_lsqr] = blendenpik(A2, b2, gamma, "lsqr", "DCT", ... 
                                        tol, maxit, verbose);
                   
figure(2);
semilogy(1:length(resvecA1_mres), resvecA1_mres);
hold on;
semilogy(1:length(resvecA1_lsqr), resvecA1_lsqr);
title("A1 Iterative Solver Residuals (\gamma = " + gamma + ")");
legend("A1 Blendenpik with MinRes", "A1 Blendenpik with LSQR");
xlabel("Iteration");
ylabel("$||\tilde{A}^{T}r_{i}||_{2}$", 'Interpreter', 'Latex');
grid on;
hold off;

figure(3);
semilogy(1:length(resvecA2_mres), resvecA2_mres);
hold on;
semilogy(1:length(resvecA2_lsqr), resvecA2_lsqr);
title("A2 Iterative Solver Residuals (\gamma = " + gamma + ")");
legend("A2 Blendenpik with MinRes", "A2 Blendenpik with LSQR");
xlabel("Iteration");
ylabel("$||\tilde{A}^{T}r_{i}||_{2}$", 'Interpreter', 'Latex');
grid on;
hold off;
