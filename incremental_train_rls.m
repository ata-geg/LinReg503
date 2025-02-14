function [w, b] = incremental_train_rls(X, y, lambda, epsilon)
    [m, d] = size(X);
    
    % Initialize the inverse of the regularization matrix Q
    Q_inv_diag = [1/epsilon; (1/lambda)*ones(d, 1)];
    C_inv = diag(Q_inv_diag);
    rhs = zeros(d+1, 1);
    
    for i = 1:m
        x_i = X(i, :)';
        x_tilde = [1; x_i]; % Augment with bias term
        
        % Sherman-Morrison update for C_inv
        u = x_tilde;
        v = x_tilde;
        numerator = C_inv * (u * v') * C_inv;
        denominator = 1 + v' * C_inv * u;
        C_inv = C_inv - numerator / denominator;
        

        rhs = rhs + x_tilde * y(i);
    end
    
    % Compute the optimal weights
    w_tilde = C_inv * rhs;
    b = w_tilde(1);
    w = w_tilde(2:end);
end