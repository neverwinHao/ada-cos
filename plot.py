import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 定义五个CSV文件的文件名列表
csv_files = ['Ours.csv', 'MRRNet.csv', 'SPARNet.csv', 'SRGAN.csv', 'UFSRNet.csv']

# 定义一系列相似度阈值
# thresholds = [0.75, 0.8, 0.85, 0.90, 0.95]
thresholds = np.linspace(0.7, 0.95, 20)
# 存储每个CSV文件的图像比例结果和文件名
all_proportions = []
legend_names = []

# 遍历每个CSV文件
for csv_file in csv_files:
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 计算在不同阈值下的图像比例
    proportions = []
    for threshold in thresholds:
        proportion = (df['Similarity'] >= threshold).mean()
        proportions.append(proportion)
    
    # 存储结果和文件名
    all_proportions.append(proportions)
    legend_names.append(csv_file[:-4])  # 去除.csv扩展名


for i, proportions in enumerate(all_proportions):
    # plt.plot(thresholds, proportions, marker='o', label=legend_names[i])
    plt.plot(thresholds, proportions, label=legend_names[i])

plt.xlabel('Cosine Similarity')
plt.ylabel('Proportion of Test Images')
# plt.title('Proportion of Test Images vs. Cosine Similarity Threshold')
plt.legend()
# plt.grid(True)
plt.show()
