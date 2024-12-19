clear;
clc;
close all;

% Load all data
for nc = 1:10
    T_Acc_Data_TDFD_Day1 = load(sprintf('U%02d_Acc_TimeD_FreqD_FDay.mat', nc));
    T_Acc_Data_TDFD_Day2 = load(sprintf('U%02d_Acc_TimeD_FreqD_MDay.mat', nc));

    Temp_Acc_Data_TDFD_Day1 = T_Acc_Data_TDFD_Day1.Acc_TDFD_Feat_Vec(1:36, 1:131);
    Temp_Acc_Data_TDFD_Day2 = T_Acc_Data_TDFD_Day2.Acc_TDFD_Feat_Vec(1:36, 1:131);

    Acc_TDFD_Data_Day1{nc} = Temp_Acc_Data_TDFD_Day1;
    Acc_TDFD_Data_Day2{nc} = Temp_Acc_Data_TDFD_Day2;
end

% this will store values for each user
all_accuracies = zeros(10, 1);
all_eer_values = zeros(10, 1);
all_far_values = zeros(10, 1); % Store FAR values for each user
all_frr_values = zeros(10, 1); % Store FRR values for each user
all_auc_values = zeros(10, 1); % Store AUC values for each user
all_fpr_values = cell(10, 1);  % Store FPR values for each user
all_tpr_values = cell(10, 1);  % Store TPR values for each user


for checking_user = 1:10
    % making training and testing datasets
    training_features = [];
    testing_features = [];
    training_labels = [];
    testing_labels = [];

    % this will make a seperate dataset for the checking user
    checking_user_feat = [Acc_TDFD_Data_Day1{checking_user}; Acc_TDFD_Data_Day2{checking_user}];
    total_features = size(checking_user_feat, 1);

    % this will choose 75% of the samples for training and 25% for testing
    checking_user_train_feat = round(0.75 * total_features);
    checking_user_test_feat = total_features - checking_user_train_feat;

     % this will shuffle the rows and ramndomize them for splitting
    shuffled_rows_checking = randperm(total_features);
    checking_user_train_set = checking_user_feat(shuffled_rows_checking(1:checking_user_train_feat), :);
    checking_user_test_set = checking_user_feat(shuffled_rows_checking(checking_user_train_feat+1:end), :);

    % this will store non checking users data
    non_checking_train_set = [];
    non_checking_test_set = [];

    % below code is for taking non checking users
    for nc = 1:10
        if nc ~= checking_user
            non_checking_user_feat = [Acc_TDFD_Data_Day1{nc}; Acc_TDFD_Data_Day2{nc}];
            total_rows_non_checking = size(non_checking_user_feat, 1);

            non_checking_train_feat = round(0.75 * total_rows_non_checking);
            non_checking_test_feat = total_rows_non_checking - non_checking_train_feat;

            indices_user = randperm(total_rows_non_checking);

            non_checking_train_set = [non_checking_train_set; non_checking_user_feat(indices_user(1:non_checking_train_feat), :)];
            non_checking_test_set = [non_checking_test_set; non_checking_user_feat(indices_user(non_checking_train_feat+1:end), :)];
        end
    end

    %taking same amount of samples for non checking randomly as the checking user
    indices_train = randperm(size(non_checking_train_set, 1), checking_user_train_feat);
    selected_non_checking_train = non_checking_train_set(indices_train, :);

    %taking same amount of samples for non checking randomly as the checking user
    indices_test = randperm(size(non_checking_test_set, 1), checking_user_test_feat);
    selected_non_checking_test = non_checking_test_set(indices_test, :);

    % Combine
    full_training_features = [checking_user_train_set; selected_non_checking_train];
    full_training_labels = [ones(size(checking_user_train_set, 1), 1); zeros(size(selected_non_checking_train, 1), 1)];

    % Combine
    testing_features = [checking_user_test_set; selected_non_checking_test];
    testing_labels = [ones(size(checking_user_test_set, 1), 1); zeros(size(selected_non_checking_test, 1), 1)];

    % hiddenlayers
    hiddenLayerSize = [20, 10];
    net = feedforwardnet(hiddenLayerSize);

    % this code is for activation functions
    net.layers{1}.transferFcn = 'logsig';
    net.layers{2}.transferFcn = 'tansig';
    net.layers{end}.transferFcn = 'purelin';

    % this code is for weights and biases
    for i = 1:numel(net.layers)
        net.IW{1, 1} = randn(size(net.IW{1, 1})) * 0.01;
        net.b{1} = randn(size(net.b{1})) * 0.01;
        if i < numel(net.layers)
            net.LW{i+1, i} = randn(size(net.LW{i+1, i})) * 0.01;
            net.b{i+1} = randn(size(net.b{i+1})) * 0.01;
        end
    end

    % training
    net.trainParam.epochs = 1000;
    net.trainParam.goal = 1e-6;

    % this will train the neural network
    net = train(net, full_training_features', full_training_labels');

    % predictions
    network_predictions = net(testing_features');
    predicted_labels = network_predictions > 0.5;
    all_predictions{checking_user} = predicted_labels';

    % this will calculate the accuracy
    all_accuracies(checking_user) = sum(predicted_labels' == testing_labels) / length(testing_labels) * 100;

    % below code will calculate the FAR and FRR using predicted labels and
    % testing labels
    false_acceptances = sum((predicted_labels' == 1) & (testing_labels == 0));
    false_rejections = sum((predicted_labels' == 0) & (testing_labels == 1));

    non_checking_attempts = sum(testing_labels == 0);
    checking_attempts = sum(testing_labels == 1);

    all_far_values(checking_user) = false_acceptances / non_checking_attempts;
    all_frr_values(checking_user) = false_rejections / checking_attempts;

    % this will calculate the ROC and AUC 
    [FPR, TPR, ~, AUC] = perfcurve(testing_labels, network_predictions, 1);
    all_fpr_values{checking_user} = FPR;
    all_tpr_values{checking_user} = TPR;
    all_auc_values(checking_user) = AUC;

    %EER
    diff = abs(FPR - (1 - TPR));
    eer_index = find(diff == min(diff), 1);
    all_eer_values(checking_user) = (FPR(eer_index) + (1 - TPR(eer_index))) / 2 * 100;

    % this will plot ROC Curve
    figure;
    plot(FPR, TPR, 'b-', 'LineWidth', 2);
    hold on;
    plot([0 1], [0 1], 'k--');
    plot(FPR(eer_index), TPR(eer_index), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(['ROC Curve for User ', num2str(checking_user)]);
    text(FPR(eer_index), TPR(eer_index), sprintf(' EER = %.2f%%', all_eer_values(checking_user)), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
    grid on;
end

% this will pripnt the below mentioned values in the command window
fprintf('User-wise Accuracy, EER, FAR, FRR, and AUC:\n');
for user = 1:10
    fprintf('User %d - Accuracy: %.2f%%, EER: %.2f%%, FAR: %.4f, FRR: %.4f, AUC: %.4f\n', ...
        user, all_accuracies(user), all_eer_values(user), all_far_values(user), all_frr_values(user), all_auc_values(user));
end

% this will display the averages of these values after calculating
average_accuracy = mean(all_accuracies);
average_eer = mean(all_eer_values);
average_far = mean(all_far_values);
average_frr = mean(all_frr_values);
average_auc = mean(all_auc_values);
average_fpr = mean(cellfun(@mean, all_fpr_values));
average_tpr = mean(cellfun(@mean, all_tpr_values));

%this will print the average outputs of the values we get
% average values foraccuracy,EER,FAR,FRR,AUC,FPR,TPR
fprintf('\nOverall Results:\n');
fprintf('Average Accuracy: %.2f%%\n', average_accuracy);
fprintf('Average EER: %.2f%%\n', mean(all_eer_values));
fprintf('Average FAR: %.4f\n', average_far);
fprintf('Average FRR: %.4f\n', average_frr);
fprintf('Average AUC: %.4f\n', average_auc);
fprintf('Average FPR: %.4f\n', average_fpr);
fprintf('Average TPR: %.4f\n', average_tpr);