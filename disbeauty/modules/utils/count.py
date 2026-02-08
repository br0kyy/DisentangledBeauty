import pandas as pd
import os

# 读取CSV文件
df = pd.read_csv(r"Experiment/Results/ffhq_3000/enhancement_effects.csv")

# 过滤出rate=0的数据
rate_0_df = df[df['rate'] == 0]

# 计算id_sim大于0.75的图片数量
high_id_sim_count = len(rate_0_df[rate_0_df['id_sim'] > 0.75])

# 获取总图片数
total_images = len(rate_0_df)

print(f"Rate=0时id_sim大于0.75的图片数量: {high_id_sim_count}")
print(f"总图片数: {total_images}")
print(f"占比: {(high_id_sim_count/total_images)*100:.2f}%")

# 获取rate=0且id_sim>0.75的图片路径列表
qualified_images = rate_0_df[rate_0_df['id_sim'] > 0.75]['image_path'].unique()
print(f"符合条件的独立图片数量: {len(qualified_images)}")

# 创建一个新的DataFrame，包含这些图片在所有rate下的数据
filtered_df = df[df['image_path'].isin(qualified_images)]

# 检查数据
print(f"筛选后的总行数: {len(filtered_df)}")
print(f"包含的不同rate值: {filtered_df['rate'].unique()}")
print(f"平均每张图片的行数: {len(filtered_df) / len(qualified_images):.1f}")

# 确保输出目录存在
output_dir = os.path.dirname(r"Experiment/Results/ffhq_3000/id_sim_0.75.csv")
os.makedirs(output_dir, exist_ok=True)

# 将结果保存到新的CSV文件
filtered_df.to_csv(r"Experiment/Results/ffhq_3000/id_sim_0.75.csv", index=False)

print(f"已将符合条件的图片数据导出到: Experiment/Results/ffhq_3000/id_sim_0.75.csv")