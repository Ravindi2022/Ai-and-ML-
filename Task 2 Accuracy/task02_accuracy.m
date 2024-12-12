for nc = 1:10
    T_Acc_Data_TD_Day1 = load(sprintf('U%02d_Acc_TimeD_FDay.mat', nc));
    T_Acc_Data_TD_Day2 = load(sprintf('U%02d_Acc_TimeD_MDay.mat', nc));

    Temp_Acc_Data_TD_Day1 = T_Acc_Data_TD_Day1.Acc_TD_Feat_Vec(1:36, 1:88);
    Temp_Acc_Data_TD_Day2 = T_Acc_Data_TD_Day2.Acc_TD_Feat_Vec(1:36, 1:88);

    Acc_TD_Data_Day1{nc} = Temp_Acc_Data_TD_Day1;
    Acc_TD_Data_Day2{nc} = Temp_Acc_Data_TD_Day2;
end



% Initialize training and testing datasets
Training_Data = [];
Testing_Data = [];
Training_Labels = [];
Testing_Labels = [];

% Separate datasets for the checking user
U01 = [Acc_TD_Data_Day1{1}; Acc_TD_Data_Day2{1}];
num_samples = size(U01, 1);

% Split checking user data into 75% training and 25% testing
train_size_checking = round(0.75 * num_samples); % 54 samples
test_size_checking = num_samples - train_size_checking; % 18 samples

% Randomize rows for splitting
rng(42); % For reproducibility
indices_checking = randperm(num_samples);
checking_user_train = U01(indices_checking(1:train_size_checking), :);
checking_user_test = U01(indices_checking(train_size_checking+1:end), :);

% Initialize non-checking users' combined data
non_checking_train = [];
non_checking_test = [];

% Process non-checking users
for nc = 1:10
    if nc ~= 1
        % Combine data for the non-checking user
        user_data = [Acc_TD_Data_Day1{nc}; Acc_TD_Data_Day2{nc}];
        num_samples_user = size(user_data, 1);
        
        % Split into 75% training and 25% testing
        train_size_user = round(0.75 * num_samples_user); % 54 samples
        test_size_user = num_samples_user - train_size_user; % 18 samples
        
        % Randomize rows for splitting
        rng(42 + nc); % Ensure reproducibility
        indices_user = randperm(num_samples_user);
        
        % Take required amount for training and testing
        non_checking_train = [non_checking_train; user_data(indices_user(1:train_size_user), :)];
        non_checking_test = [non_checking_test; user_data(indices_user(train_size_user+1:end), :)];
    end
end

% Randomly select 54 samples from non-checking users' training data
rng(42); % For reproducibility
indices_train = randperm(size(non_checking_train, 1), train_size_checking); % 54 samples
selected_non_checking_train = non_checking_train(indices_train, :);

% Randomly select 18 samples from non-checking users' testing data
rng(42); % For reproducibility
indices_test = randperm(size(non_checking_test, 1), test_size_checking); % 18 samples
selected_non_checking_test = non_checking_test(indices_test, :);

% Combine training dataset
Training_Data = [checking_user_train; selected_non_checking_train];
Training_Labels = [ones(size(checking_user_train, 1), 1); zeros(size(selected_non_checking_train, 1), 1)]; % Checking user labeled as 1, non-checking as 0

% Combine testing dataset
Testing_Data = [checking_user_test; selected_non_checking_test];
Testing_Labels = [ones(size(checking_user_test, 1), 1); zeros(size(selected_non_checking_test, 1), 1)]; % Checking user labeled as 1, non-checking as 0

% Display summaries
fprintf('Checking User Training Data: %d samples\n', size(checking_user_train, 1));
fprintf('Checking User Testing Data: %d samples\n', size(checking_user_test, 1));
fprintf('Non-Checking Users Training Data: %d samples\n', size(selected_non_checking_train, 1));
fprintf('Non-Checking Users Testing Data: %d samples\n', size(selected_non_checking_test, 1));
fprintf('Total Training Data: %d samples\n', size(Training_Data, 1));
fprintf('Total Testing Data: %d samples\n', size(Testing_Data, 1));


% Adjust labels to start from 1 for dummyvar compatibility
Adjusted_Training_Labels = Training_Labels + 1;
Adjusted_Testing_Labels = Testing_Labels + 1;

% Convert labels to one-hot encoding
Training_Labels_OneHot = dummyvar(Adjusted_Training_Labels); % Convert to one-hot

% Define the neural network
hiddenLayerSize = 10; % 10 hidden layers
net = feedforwardnet(hiddenLayerSize);

% Configure training parameters
net.trainParam.epochs = 1000;  % Maximum number of epochs
net.trainParam.goal = 1e-4;    % Performance goal
net.trainParam.max_fail = 6;   % Maximum validation failures
net.trainParam.lr = 0.01;      % Learning rate

% Train the neural network
net = train(net, Training_Data', Training_Labels_OneHot');

% Get predictions on testing data
predictions = net(Testing_Data'); % Predict on test data

% Convert predictions from one-hot encoding to class labels
[~, predicted_labels] = max(predictions, [], 1); % Predicted class indices

% Adjust predicted labels back to original range (subtract 1)
predicted_labels = predicted_labels - 1;

% Calculate accuracy
accuracy = sum(predicted_labels' == Testing_Labels) / length(Testing_Labels) * 100;
fprintf('Test Accuracy: %.2f%%\n', accuracy);