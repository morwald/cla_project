function [x, iter, resvec] = blendenpik(A, b, gamma, iter_method, ... 
                           transform_type, tol, maxit, verbose)
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
        
%         SM = S * M;
        SM = M(find(S_inds == 1), :);
        [Q, R] = qr(SM, 0);
        
        if verbose == true
            fprintf("cond(inv(R')*A'*A*inv(R)): %f\n", ..., 
                cond(inv(R') * (A' * A) * inv(R)));
            % cond(R' \ ((A' * A) / R))
        end
        cond_estimate = rcond(R);
        if 1 / cond_estimate > 5 * eps(1)
            if iter_method == "minres"
                B = A / R;
                [x, flag, relres, iter, resvec] = minres(B'*B, B'*b, ...
                                                         tol, maxit);
                x = R \ x;
            elseif iter_method == "lsqr"
                B = A / R; 
                [x, flag, relres, iter, resvec] = lsqr(B'*B, B'*b, ..., 
                                                       tol, maxit);
                x = R \ x;
            else
                fprintf("Enter a valid iterative method (minres or lsqr)");
                return;
            end
            return;
        elseif n_iter > 3
            fprintf("Failure, solve using LAPACK \n");
            x = A \ b;
            return
        end
    end
end