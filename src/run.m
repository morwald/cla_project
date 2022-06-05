%% Defining Matrices
clear all; close all; clc;
tol = 1e-12; maxit = 200; verbose=false; 

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
gammas = 2:1:16; n_trials = 5;

% i) Incoherent, ill-conditioned matrix 
fprintf("Incoherent, ill-conditioned matrix\n");
fprintf("cond(A1): %d\n", cond(A1));
fprintf("cond(A1'*A1): %d\n", cond(A1'*A1));

errorsA1 = [];
itersA1 = [];
dct_timesA1 = zeros(n_trials, length(gammas));
qr_timesA1 = zeros(n_trials, length(gammas));
iter_timesA1 = zeros(n_trials, length(gammas));

for n = 1:n_trials
    fprintf("Trial: %d\n", n);
    for i = 1:length(gammas)
        gamma = gammas(i);
        fprintf("gamma: %d\n", gamma);
        [x_tilde, iters, resvecA1, time] = blendenpik(A1, b1, gamma, "minres", ...
                                                "DCT", tol, maxit, verbose);
        if n == 1
            errorsA1 = [errorsA1, norm(x1 - x_tilde)];
            itersA1 = [itersA1, iters];
        end 
        dct_timesA1(n, i) = time.dct;
        qr_timesA1(n, i) = time.qr;
        iter_timesA1(n, i) = time.iter_method;
    end
end
fprintf("\n\n");

% ii) Coherent, ill-conditioned matrix
fprintf("Coherent, ill-conditioned matrix\n");
fprintf("cond(A2): %d \n", cond(A2));
fprintf("cond(A2'*A2): %d \n", cond(A2'*A2));

errorsA2 = [];
itersA2 = [];
dct_timesA2 = zeros(n_trials, length(gammas));
qr_timesA2 = zeros(n_trials, length(gammas));
iter_timesA2 = zeros(n_trials, length(gammas));
for n = 1:n_trials
    fprintf("Trial: %d\n", n);
    for i = 1:length(gammas)
        gamma = gammas(i);
        fprintf("gamma: %d\n", gamma);
        [x_tilde, iters, resvecA2, time] = blendenpik(A2, b2, gamma, "minres", ...
                                                "DCT", tol, maxit, verbose);
        
        if n == 1
            errorsA2 = [errorsA2, norm(x2 - x_tilde)];
            itersA2 = [itersA2, iters];
        end 
        dct_timesA2(n, i) = time.dct;
        qr_timesA2(n, i) = time.qr;
        iter_timesA2(n, i) = time.iter_method;
    end
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

figure(2);
plot(gammas, dct_timesA1(1, 1:end) + qr_timesA1(1, 1:end) + iter_timesA1(1, 1:end), '-*');
hold on;
plot(gammas, dct_timesA2(1, 1:end) + qr_timesA2(1, 1:end) + iter_timesA2(1, 1:end), '-o');
title("A1 and A2 convergence vs \gamma (tol=" + tol + ")");
legend("A1 MinRes Times in Blendenpik", "A2 MinRes Times in Blendenpik");
xlabel("\gamma");
ylabel("Time [s]");
grid on;
hold off;

figure(3);
plot(gammas, mean(dct_timesA1, 1), '--');
hold on;
plot(gammas, mean(qr_timesA1, 1), '--');
plot(gammas, mean(iter_timesA1, 1), '--');
plot(gammas, mean(dct_timesA1 + qr_timesA1 + iter_timesA1, 1), '-*', 'LineWidth', 2);
plot(gammas, mean(dct_timesA2, 1), '-.');
plot(gammas, mean(qr_timesA2, 1), '-.');
plot(gammas, mean(iter_timesA2, 1), '-.');
plot(gammas, mean(dct_timesA2 + qr_timesA2 + iter_timesA2, 1), '-o', 'LineWidth', 2);
title("A1 and A2 convergence vs \gamma averaged over " + n_trials + " runs (tol=" + tol + ")");
legend("A1 DCT Time", "A1 QR time", "A1 Inner MINRES time", "A1 Total Time", ...
    "A2 DCT Time", "A2 QR time", "A2 Inner MINRES time", "A2 Total Time");
xlabel("\gamma");
ylabel("Time [s]");
grid on;
hold off;

%% part f) Convergence of inner LSQR/ MINRES steps in Blendenpik
gamma = 6; maxit = 200;

[x1m, ~, resvecA1_mres, ~] = blendenpik(A1, b1, gamma, "minres", "DCT", ... 
                                        tol, maxit, verbose);
[x1l, ~, resvecA1_lsqr, ~] = blendenpik(A1, b1, gamma, "lsqr", "DCT", ... 
                                        tol, maxit, verbose);

[x2m, ~, resvecA2_mres, ~] = blendenpik(A2, b2, gamma, "minres", "DCT", ... 
                                        tol, maxit, verbose);
[x2l, ~, resvecA2_lsqr, ~] = blendenpik(A2, b2, gamma, "lsqr", "DCT", ... 
                                        tol, maxit, verbose);
                   
figure(4);
semilogy(1:length(resvecA1_mres), resvecA1_mres);
hold on;
semilogy(1:length(resvecA1_lsqr), resvecA1_lsqr);
title("A1 Iterative Solver Residuals (\gamma = " + gamma + ", tol = " + tol + ")");
legend("A1 Blendenpik with MinRes", "A1 Blendenpik with LSQR");
xlabel("Iteration");
ylabel("$||\tilde{A}^{T}r_{i}||_{2}$", 'Interpreter', 'Latex');
grid on;
hold off;

figure(5);
semilogy(1:length(resvecA2_mres), resvecA2_mres);
hold on;
semilogy(1:length(resvecA2_lsqr), resvecA2_lsqr);
title("A2 Iterative Solver Residuals (\gamma = " + gamma + ", tol = " + tol + ")");
legend("A2 Blendenpik with MinRes", "A2 Blendenpik with LSQR");
xlabel("Iteration");
ylabel("$||\tilde{A}^{T}r_{i}||_{2}$", 'Interpreter', 'Latex');
grid on;
hold off;
