clear all; close all; clc;
gammas = 2:1:16; tol = 1e-12; maxit = 200;

% e) i)
rng(11);
U = orth(rand(20000, 400));
S = diag(linspace(1, 1e5, 400));
V = orth(rand(400));
A1 = U*S*V';
b1 = rand(size(A1, 1), 1);
x1 = A1 \ b1;

fprintf("Incoherent, ill-conditioned matrix\n");
fprintf("cond(A1): %d\n", cond(A1));
fprintf("cond(A1'*A1): %d\n", cond(A1'*A1));

errorsA1 = [];
itersA1 = [];
for gamma = gammas
    fprintf("gamma: %d\n", gamma);
    [x_tilde, iters] = blendenpik(A1, b1, gamma, "DCT");
    
    errorsA1 = [errorsA1, norm(x1 - x_tilde)];
    itersA1 = [itersA1, iters];
end
[x, flag, relres, iters_no_pc_min_res_A1] = minres(A1' * A1, A1' * b1,  tol, maxit, []);
fprintf("\n\n");

% e) ii)
rng(11);
A2 = [ diag(linspace(1,1e5,400)); zeros(19600,400) ];
A2 = A2 + 1e-8*ones(20000,400);
b2 = rand(size(A2, 1), 1);
x2 = A2 \ b2;

fprintf("Coherent, ill-conditioned matrix\n");
fprintf("cond(A2): %d \n", cond(A2));
fprintf("cond(A2'*A2): %d \n", cond(A2'*A2));

errorsA2 = [];
itersA2 = [];
for gamma = gammas
    fprintf("gamma: %d\n", gamma);
    [x_tilde, iters] = blendenpik(A2, b2, gamma, "DCT");
    
    errorsA2 = [errorsA2, norm(x2 - x_tilde)];
    itersA2 = [itersA2, iters];
end

[x, flag, relres, iters_no_pc_min_res_A2] = minres(A2' * A2, A2' * b2,  tol, maxit, []);
fprintf("\n\n");

% random matrix with low coherence
rng(11);
m = 20000; n = 100;
A3 = rand(m, n);
b3 = rand(m, 1);
x3 = A3 \ b3;

% fprintf("Random matrix\n");
% fprintf("cond(A3): %d \n", cond(A3));
% fprintf("cond(A3'*A3): %d \n", cond(A3'*A3));
% 
% errorsA3 = [];
% itersA3 = [];
% for gamma = gammas
%     fprintf("gamma: %d\n", gamma);
%     [x_tilde, iters] = blendenpik(A3, b3, gamma, "DCT");
%     
%     errorsA3 = [errorsA3, norm(x3 - x_tilde)];
%     itersA3 = [itersA3, iters];
% end
% [x, flag, relres, iters_no_pc_min_res_A3] = minres(A3' * A3, A3' * b3,  tol, maxit, []);

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

% figure(2);
% plot(gammas, itersA1);
% hold on;
% plot(gammas, iters_no_pc_min_res_A1 * ones(length(gammas)));
% title("Incoherent, ill-conditioned matrix A1 convergence (tol=" + tol + ")");
% legend("MinRes Iterations in Blendenpik", "MinRes Iterations no preconditioner");
% xlabel("Gamma values");
% ylabel("Number of iterations to convergence");
% ylim([0, maxit * 1.1]);
% grid on;
% hold off;

% figure(3);
% plot(gammas, itersA2);
% hold on;
% plot(gammas, iters_no_pc_min_res_A2 * ones(length(gammas)));
% title("Coherent, ill-conditioned matrix A2 convergence (tol=" + tol + ")");
% legend("MinRes Iterations in Blendenpik", "MinRes Iterations no preconditioner");
% xlabel("Gamma values");
% ylabel("Number of iterations to convergence");
% ylim([0, maxit * 1.1]);
% grid on;
% hold off;

% figure(4);
% plot(gammas, itersA3);
% hold on;
% plot(gammas, iters_no_pc_min_res_A3 * ones(length(gammas)));
% title("Random matrix A3 convergence (tol=" + tol + ")");
% legend("MinRes Iterations in Blendenpik", "MinRes Iterations no preconditioner");
% xlabel("Gamma values");
% ylabel("Number of iterations to convergence");
% ylim([0, maxit * 1.1]);
% grid on;
% hold off;
