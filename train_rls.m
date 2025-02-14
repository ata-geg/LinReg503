%Ask for function input
prompt = "What is your X? ";

X = input(prompt);

prompt1 = "What is your y? ";

y = input(prompt1);

prompt2 = "What is your lambda? ";

lambda = input(prompt2);

prompt3 = "What is your epsilon? ";

epsilon = input(prompt3);

function [w, b] = train_rls(X, y, lambda, epsilon)
    [m, d] = size(X);

    % Augment the design matrix with a column of ones for the bias term
    X_tilde = [ones(m, 1), X];

    Q = zeros(d+1, d+1);
    Q(1, 1) = epsilon;
    Q(2:end, 2:end) = lambda * eye(d);

    % Compute C and the right-hand side of the equation
    C = X_tilde' * X_tilde + Q;
    rhs = X_tilde' * y;
    
    % Solve for w_tilde using pseudoinverse to handle non-invertible matrices
    w_tilde = pinv(C) * rhs;
    b = w_tilde(1);
    w = w_tilde(2:end);
end