function [x, iter] = blendenpik(A, b, gamma, transform_type, tol, verbose)
    [m, n] = size(A);
    
    if m < n
        fprintf("Error, we only consider overdetermined systems\n")
        return;
    end
    
    if transform_type == "WHT"
        m_tilde = 2 ^ (ceil(log2(m)));
    elseif transform_type == "DCT" || transform_type == "DHT"
        m_tilde = ceil(m / 1000) * 1000;
    else
        fprintf("Enter a valid transform type");
        return
    end
    
    M = [A; zeros(m_tilde - m, n)];
    
    n_iter = 0;
    while true
        n_iter = n_iter + 1;
        D = randi([0, 1], [m_tilde, 1]) * 2 - 1;
        D = diag(D);
        M = dct(D * M);
        
        S = rand([m_tilde, 1]);
        S_inds = S <= gamma * n / m_tilde;
        S(S_inds) =  1;
        S(~S_inds) = 0;
        S = diag(S);
        
        R = qr(S * M);
        R = triu(R(1:n, 1:n));
        
        if verbose == true
            fprintf("cond(inv(R')*A'*A*inv(R)): %f\n", ..., 
                cond(inv(R') * (A' * A) * inv(R)));
        end
        cond_estimate = rcond(R);
        if 1 / cond_estimate > 5 * eps(1)
            [x, flag, relres, iter] = minres(A'* A, A'* b, tol, 100, R', R);
%             x = minres(A'* A, A'* b, 10^-14, 100, R' * R);
%             x = minres(A'* A, A'* b, 10^-8, 100, []);
%             x = lsqr(A, b, 10^-14, 100, R);
%             x = lsqr(A, b, 10^-14, 100, []);
            return
        elseif n_iter > 3
            fprintf("Failure, solve using LAPACK \n");
            x = A \ b;
            return
        end
    end
end