% Load and preprocess data
data = load('arrhythmia.mat');
X = data.X;
Y = data.Y;

% Convert labels from {0, 1} to {-1, 1}
Y = 2*Y - 1;

% Impute missing values (NaNs) with column median. Handle all-NaN columns.
[n, d] = size(X);
for col = 1:d
    col_data = X(:, col);
    col_median = median(col_data, 'omitnan');
    
    % If all values are NaN, replace with 0
    if isnan(col_median)
        col_median = 0;
    end
    
    col_data(isnan(col_data)) = col_median;
    X(:, col) = col_data;
end

% Verify no NaNs remain
assert(~any(isnan(X(:))), 'NaNs still present in X after imputation');

% Split into training (80%) and testing (20%)
rng('default');
shuffled_idx = randperm(n);
split = round(0.8*n);
X_train = X(shuffled_idx(1:split), :);
Y_train = Y(shuffled_idx(1:split));
X_test = X(shuffled_idx(split+1:end), :);
Y_test = Y(shuffled_idx(split+1:end));

% Standardize training and test data (handle zero variance)
mu = mean(X_train, 1);
sigma = std(X_train, [], 1);
sigma(sigma == 0) = 1;  % Avoid division by zero for constant features

X_train = (X_train - mu) ./ sigma;
X_test = (X_test - mu) ./ sigma;

% Define hyperparameter grids (log scale)
lambda_grid = 2.^(-5:2:5);
gamma_grid = 2.^(-5:2:5);
degree_grid = [1, 2, 3];

% ... Rest of the code (3-fold CV for SVM training) remains the same ...

% Initialize best parameters and accuracy storage
best = struct('linear', [], 'gaussian', [], 'poly', []);
test_acc = struct('linear', 0, 'gaussian', 0, 'poly', 0);

%% Linear SVM
fprintf('Training Linear SVM...\n');
cv = cvpartition(Y_train, 'KFold', 3);
acc_linear = zeros(length(lambda_grid), 1);

for i = 1:length(lambda_grid)
    C = 1 / (2 * lambda_grid(i)); % C = 1/(2λ)
    acc = zeros(3, 1);
    for fold = 1:3
        trainIdx = cv.training(fold);
        valIdx = cv.test(fold);
        mdl = fitcsvm(X_train(trainIdx, :), Y_train(trainIdx), ...
            'KernelFunction', 'linear', 'BoxConstraint', C, 'Standardize', false);
        pred = predict(mdl, X_train(valIdx, :));
        acc(fold) = sum(pred == Y_train(valIdx)) / numel(valIdx);
    end
    acc_linear(i) = mean(acc);
end

[~, idx] = max(acc_linear);
best.linear.C = 1 / (2 * lambda_grid(idx));
best.linear.lambda = lambda_grid(idx);

% Train final linear model
mdl_linear = fitcsvm(X_train, Y_train, 'KernelFunction', 'linear', ...
    'BoxConstraint', best.linear.C, 'Standardize', false);
test_acc.linear = sum(predict(mdl_linear, X_test) == Y_test) / numel(Y_test);

%% Gaussian (RBF) SVM
fprintf('Training Gaussian SVM...\n');
acc_gaussian = zeros(length(lambda_grid), length(gamma_grid));

for i = 1:length(lambda_grid)
    C = 1 / (2 * lambda_grid(i));
    for j = 1:length(gamma_grid)
        sigma = 1 / sqrt(2 * gamma_grid(j)); % KernelScale = 1/sqrt(2γ)
        acc = zeros(3, 1);
        for fold = 1:3
            trainIdx = cv.training(fold);
            valIdx = cv.test(fold);
            mdl = fitcsvm(X_train(trainIdx, :), Y_train(trainIdx), ...
                'KernelFunction', 'rbf', 'BoxConstraint', C, ...
                'KernelScale', sigma, 'Standardize', false);
            pred = predict(mdl, X_train(valIdx, :));
            acc(fold) = sum(pred == Y_train(valIdx)) / numel(valIdx);
        end
        acc_gaussian(i, j) = mean(acc);
    end
end

[max_acc, idx] = max(acc_gaussian(:));
[i, j] = ind2sub(size(acc_gaussian), idx);
best.gaussian.C = 1 / (2 * lambda_grid(i));
best.gaussian.lambda = lambda_grid(i);
best.gaussian.gamma = gamma_grid(j);

% Train final Gaussian model
mdl_gaussian = fitcsvm(X_train, Y_train, 'KernelFunction', 'rbf', ...
    'BoxConstraint', best.gaussian.C, ...
    'KernelScale', 1/sqrt(2*best.gaussian.gamma), 'Standardize', false);
test_acc.gaussian = sum(predict(mdl_gaussian, X_test) == Y_test) / numel(Y_test);

%% Polynomial SVM
fprintf('Training Polynomial SVM...\n');
acc_poly = zeros(length(lambda_grid), length(degree_grid));

for i = 1:length(lambda_grid)
    C = 1 / (2 * lambda_grid(i));
    for j = 1:length(degree_grid)
        acc = zeros(3, 1);
        for fold = 1:3
            trainIdx = cv.training(fold);
            valIdx = cv.test(fold);
            mdl = fitcsvm(X_train(trainIdx, :), Y_train(trainIdx), ...
                'KernelFunction', 'polynomial', 'BoxConstraint', C, ...
                'PolynomialOrder', degree_grid(j), 'Standardize', false);
            pred = predict(mdl, X_train(valIdx, :));
            acc(fold) = sum(pred == Y_train(valIdx)) / numel(valIdx);
        end
        acc_poly(i, j) = mean(acc);
    end
end

[max_acc, idx] = max(acc_poly(:));
[i, j] = ind2sub(size(acc_poly), idx);
best.poly.C = 1 / (2 * lambda_grid(i));
best.poly.lambda = lambda_grid(i);
best.poly.degree = degree_grid(j);

% Train final polynomial model
mdl_poly = fitcsvm(X_train, Y_train, 'KernelFunction', 'polynomial', ...
    'BoxConstraint', best.poly.C, 'PolynomialOrder', best.poly.degree, ...
    'Standardize', false);
test_acc.poly = sum(predict(mdl_poly, X_test) == Y_test) / numel(Y_test);

%% Display results
fprintf('\nTest Accuracies:\n');
fprintf('Linear SVM: %.2f%%\n', test_acc.linear*100);
fprintf('Gaussian SVM: %.2f%%\n', test_acc.gaussian*100);
fprintf('Polynomial SVM: %.2f%%\n', test_acc.poly*100);