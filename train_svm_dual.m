function [w, b] = train_svm_dual(X, y, lambda)
    [m, ~] = size(X);  % Number of samples (m)

    % Construct the quadratic term matrix H
    Gram = X * X';                 % Gram matrix (m×m)
    Q = (y * y') .* Gram;          % Element-wise multiplication with label products
    H = Q / (2 * lambda);          % Scale by 1/(2λ)

    % Linear term vector f (negated for maximization → minimization)
    f = -ones(m, 1);

    % Equality constraint: ∑α_i y_i = 0
    Aeq = y';
    beq = 0;

    % Bounds: 0 ≤ α_i ≤ 1/m
    lb = zeros(m, 1);
    ub = (1/m) * ones(m, 1);

    % Solve the quadratic program using quadprog
    options = optimset('Display', 'off');
    alpha = quadprog(H, f, [], [], Aeq, beq, lb, ub, [], options);

    % Compute primal solution w from dual variables α
    w = (X' * (alpha .* y)) / (2 * lambda);

    % Compute bias b using support vectors with 0 < α_i < 1/m
    tolerance = 1e-5;
    upper_bound = (1/m) - tolerance;
    idx = find(alpha > tolerance & alpha < upper_bound);

    % Fallback if no such points: use all support vectors (α > 0)
    if isempty(idx)
        idx = find(alpha > tolerance);
    end

    % Compute b as the median of y_i - x_i*w for selected points
    if ~isempty(idx)
        b = median(y(idx) - X(idx, :) * w);
    else
        b = 0;  % Fallback if no support vectors
    end
end