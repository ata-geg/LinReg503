function [w, b] = train_svm_primal(X, y, lambda)
    [m, d] = size(X);  % Number of samples (m) and features (d)

    % Construct the quadratic term matrix H
    H = diag([2 * lambda * ones(d, 1); 0; zeros(m, 1)]);

    % Construct the linear term vector f
    f = [zeros(d + 1, 1); (1/m) * ones(m, 1)];

    % Construct inequality constraints A*z <= b
    A = [X .* (-y), -y, -eye(m)];  % Combine parts for w, b, and ξ
    b_vec = -ones(m, 1);  % Right-hand side of inequalities

    % Set lower bounds: w and b are unbounded, ξ ≥ 0
    lb = [-Inf(d + 1, 1); zeros(m, 1)];
    
    % Solve the quadratic program using quadprog
    options = optimset('Display', 'off');
    z = quadprog(H, f, A, b_vec, [], [], lb, [], [], options);

    % Extract weights w and bias b from the solution
    w = z(1:d);
    b = z(d + 1);
end