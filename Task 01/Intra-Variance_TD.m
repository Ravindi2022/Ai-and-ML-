all_users_intra_variance_day1 = zeros(10, 36);
all_users_intra_variance_day2 = zeros(10, 36);

for user = 1:10
    T_Acc_Data_Day1 = sprintf('U%02d_Acc_TimeD_FDay.mat', user);
    T_Acc_Data_Day2 = sprintf('U%02d_Acc_TimeD_MDay.mat', user);
    
    % this will load the Day 1 data
    load(T_Acc_Data_Day1);
    
    data_day1 = Acc_TD_Feat_Vec(1:36, 1:88); 
    % this will calculate intra user variance for Day 1
    intra_user_variance_day1 = var(data_day1, 0, 2);
    % this will store the results for Day 1
    all_users_intra_variance_day1(user, :) = intra_user_variance_day1;

    % this will load data for day 2
    load(T_Acc_Data_Day2);
    data_day2 = Acc_TD_Feat_Vec(1:36, 1:88); 
    % this calculates intra user variance
    intra_user_variance_day2 = var(data_day2, 0, 2);s
    % this will store the results for Day 2
    all_users_intra_variance_day2(user, :) = intra_user_variance_day2;
end

% this will generate a graph for day 1
figure;
hold on;
for user = 1:10
    plot(1:36, all_users_intra_variance_day1(user, :), '-o', 'LineWidth', 1.5, 'DisplayName', sprintf('User %d', user));
end
xlabel('Sample Index');
ylabel('Intra-user Variance');
title('Intra-user Variance for All Users (Day 1)');
legend('show', 'Location', 'best');
grid on;
hold off;

% this will generate a graph for day 2
figure;
hold on;
for user = 1:10
    plot(1:36, all_users_intra_variance_day2(user, :), '-o', 'LineWidth', 1.5, 'DisplayName', sprintf('User %d', user));
end
xlabel('Sample Index');
ylabel('Intra-user Variance');
title('Intra-user Variance for All Users (Day 2)');
legend('show', 'Location', 'best');
grid on;
hold off;

% this will calculate and display the average intra user variance for both days
average_intra_variance_day1 = mean(all_users_intra_variance_day1, 1);
average_intra_variance_day2 = mean(all_users_intra_variance_day2, 1);

disp('Average intra-user variance for Day 1 (across all users):');
disp(average_intra_variance_day1);

disp('Average intra-user variance for Day 2 (across all users):');
disp(average_intra_variance_day2);

% this will plot the average intra user variance for Day 1 and Day 2
figure;
plot(1:36, average_intra_variance_day1, '-o', 'LineWidth', 1.5, 'DisplayName', 'Day 1');
hold on;
plot(1:36, average_intra_variance_day2, '-o', 'LineWidth', 1.5, 'DisplayName', 'Day 2');
xlabel('Sample Index');
ylabel('Average Intra-user Variance');
title('Average Intra-user Variance Across All Users');
legend('show', 'Location', 'best');
grid on;
hold off;
