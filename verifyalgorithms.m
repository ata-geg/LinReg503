
rng(42); % Set seed

m = 100; 
d = 5;   
X = randn(m, d);
y = randn(m, 1);
lambda = 0.1;
epsilon = 1e-8;

% Compute solutions using both algorithms
[w_batch, b_batch] = train_rls(X, y, lambda, epsilon);
[w_inc, b_inc] = incremental_train_rls(X, y, lambda, epsilon);

% Compare solutions 
w_tilde_batch = [b_batch; w_batch];
w_tilde_inc = [b_inc; w_inc];

% Calculate numerical difference
difference = norm(w_tilde_batch - w_tilde_inc);
tolerance = 1e-6;%Change as needed

% Display results
fprintf('Difference between solutions: %.2e\n', difference);
if difference < tolerance
    fprintf('Verification PASSED (difference < %.0e)\n', tolerance);
else
    fprintf('Verification FAILED (difference >= %.0e)\n', tolerance);
end