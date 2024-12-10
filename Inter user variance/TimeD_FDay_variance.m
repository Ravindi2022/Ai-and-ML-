
% Preallocate an array to store the feature data for each user
Acc_Data_All_Users = cell(1, 10);

% Load and store data for each user
for nc = 1:10
    % Load each user's data file, and extract the required features
    T_Acc_Data_Day1 = load(sprintf('U%02d_Acc_TimeD_FDay.mat', nc));
    
    % Extract the specific data of size (36x88)
    Temp_Acc_Data = T_Acc_Data_Day1.Acc_TD_Feat_Vec(1:36, 1:88);
    
    % Store in the preallocated cell array
    Acc_Data_All_Users{nc} = Temp_Acc_Data;
end

% Calculate variance for each feature across all users
% Initialize an empty array to store variances for features (36x88)
inter_user_variance = zeros(36, 88);

% Loop through each feature and calculate variance across users manually
for row = 1:36
    for col = 1:88
        % Extract feature values for all users for the given row and column
        feature_values = zeros(1, 10);
        
        for nc = 1:10
            feature_values(nc) = Acc_Data_All_Users{nc}(row, col);
        end
        
        % Calculate variance for the current feature across users
        inter_user_variance(row, col) = var(feature_values);
    end
end

% Calculate the average variance across all 36 rows for each of the 88 features
average_variance_per_feature = mean(inter_user_variance, 1);

% Plot the average inter-user variance per feature
figure;
plot(1:88, average_variance_per_feature, '-o');
xlabel('Feature Index');
ylabel('Average Inter-user Variance');
title('Average Inter-user Variance for TimeD_FDay');
grid on;