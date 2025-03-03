function [w, b, average_w, average_b] = train_perceptron(X, y)
    [m, d] = size(X);
    % Augment input with a column of ones for the bias term
    X_aug = [ones(m, 1), X];
    augmented_dim = d + 1;
    
    % Initialize weights (includes bias as the first element)
    w = zeros(augmented_dim, 1);
    sum_w = zeros(augmented_dim, 1); % For averaging
    updates = 0; % Count the number of weight updates
    
    max_epochs = 100; % Prevent infinite loops for non-separable data
    
    for epoch = 1:max_epochs
        % Shuffle the data each epoch
        shuffle_idx = randperm(m);
        X_shuffled = X_aug(shuffle_idx, :);
        y_shuffled = y(shuffle_idx);
        
        mistake_made = false; % Track if any update occurs in this epoch
        
        for i = 1:m
            x_i = X_shuffled(i, :)'; % Column vector
            y_i = y_shuffled(i);
            
            % Check if the current sample is misclassified
            if y_i * (w' * x_i) <= 0
                % Update weights
                w = w + y_i * x_i;
                % Accumulate weights for averaging
                sum_w = sum_w + w;
                updates = updates + 1;
                mistake_made = true;
            end
        end
        
        % Early stopping if no mistakes in an epoch
        if ~mistake_made
            break;
        end
    end
    
    % Compute averaged weights
    if updates == 0
        average_w = w; % No updates, return initial weights
    else
        average_w = sum_w / updates;
    end
    
    % Split the augmented weights into bias (b) and weights (w)
    b = w(1);
    w = w(2:end);
    
    average_b = average_w(1);
    average_w = average_w(2:end);
end