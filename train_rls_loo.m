function [w, b, train_err, loo_err] = train_rls_loo(X, y, lambda)
    [m, d] = size(X);
    % Augment the feature matrix with a column of ones for the bias term
    X_aug = [ones(m, 1), X];
    
    % The first element (bias term) is not regularized (0), others are lambda
    Lambda = diag([0; lambda * ones(d, 1)]);
    
    % Compute the matrix C
    C = X_aug' * X_aug + Lambda;
    
    % Compute the inverse of C
    C_inv = inv(C);
    
    % Compute the optimal weights 
    w_tilde = C_inv * (X_aug' * y);
    b = w_tilde(1);
    w = w_tilde(2:end);
    
    % Compute training error 
    y_pred_train = X_aug * w_tilde;
    train_err = mean((y_pred_train - y).^2);
    
    % Compute the diagonal elements of the hat matrix efficiently
    H_diag = sum((X_aug * C_inv) .* X_aug, 2);
    
    % Compute LOOCV residuals and error
    loo_residuals = (y_pred_train - y) ./ (1 - H_diag);
    loo_err = mean(loo_residuals.^2);
end