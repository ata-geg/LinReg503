function [X_poly] = generate_poly_features(X, k)
    % Get the dimensions of the input matrix X
    [m, d] = size(X);
    % Initialize the output matrix with the original features
    X_poly = X;
    
    % Iterate over each root from 2 to k
    for r = 2:k
        % Compute the r-th root for each element in X
        X_root = X.^(1/r);
        % Append the roots as new features to X_poly
        X_poly = [X_poly, X_root];
    end
end