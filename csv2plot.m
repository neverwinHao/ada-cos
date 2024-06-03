% 定义五个CSV文件的文件名列表
csv_files = {'Ours.csv', 'MRRNet.csv', 'SPARNet.csv', 'SRGAN.csv', 'UFSRNet.csv'};

% 定义一系列相似度阈值
thresholds = linspace(0.7, 0.95, 15);

% 存储每个CSV文件的图像比例结果和文件名
all_proportions = [];
legend_names = {};

% 定义线条样式和颜色
line_styles = {'-o', '-s', '-^', '-d', '-*'};
line_colors = lines(length(csv_files));

% 遍历每个CSV文件
for i = 1:length(csv_files)
    csv_file = csv_files{i};
    
    % 读取CSV文件
    data = readtable(csv_file);
    
    % 计算在不同阈值下的图像比例
    proportions = [];
    for threshold = thresholds
        proportion = mean(data.Similarity >= threshold);
        proportions = [proportions, proportion];
    end
    
    % 存储结果和文件名
    all_proportions = [all_proportions; proportions];
    legend_names{end+1} = csv_file(1:end-4); % 去除.csv扩展名
end

% 绘制 Fig. 3，每个CSV文件对应一条曲线
figure;
hold on;
for i = 1:size(all_proportions, 1)
    plot(thresholds, all_proportions(i, :), line_styles{i}, 'Color', line_colors(i, :), ...
        'MarkerFaceColor', line_colors(i, :), 'MarkerSize', 6, 'LineWidth', 1.5, 'DisplayName', legend_names{i});
end
hold off;

% 添加轴标签和图例
xlabel('Cosine Similarity', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Proportion of Test Images', 'FontSize', 12, 'FontWeight', 'bold');
legend('show', 'Location', 'northeast', 'FontSize', 10);

% 设置图表属性
set(gca, 'FontSize', 12, 'LineWidth', 1.5, 'Box', 'on', 'GridAlpha', 0.3);
%title('Proportion of Test Images vs. Cosine Similarity Threshold', 'FontSize', 16, 'FontWeight', 'bold');

% 添加网格
% 添加虚化的网格
% 添加镂空的网格
grid on;
ax = gca;
ax.GridColor = [0.8, 0.8, 0.8];  % 设置网格线颜色为浅灰色
ax.GridAlpha = 0.3;  % 设置网格线的透明度
ax.GridLineStyle = '--';  % 设置网格线为虚线

print('Cosine', '-depsc');
