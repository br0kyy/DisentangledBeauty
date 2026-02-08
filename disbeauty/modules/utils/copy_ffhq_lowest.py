import pandas as pd
import shutil
import os
from pathlib import Path

# 设置路径
src_dir = r"C:\Programming\PyProject\Facial Aesthetic Enhancement\Datasets\ffhq256"
dst_dir = r"C:\Programming\PyProject\Facial Aesthetic Enhancement\DisentangleBeauty\Experiment\Data\ffhq_3000\3000_lowest"
csv_path = r"C:\Programming\PyProject\Facial Aesthetic Enhancement\DisentangleBeauty\ffhq.csv"

# 创建目标目录(如果不存在)
Path(dst_dir).mkdir(parents=True, exist_ok=True)

# 读取CSV文件
df = pd.read_csv(csv_path)

# 过滤分数大于2.10的图片并按分数升序排序
filtered_df = df[df['score'] > 2.40].sort_values('score')

# 获取前1000张最低分的图片
selected_images = filtered_df.head(3000)

# 拷贝图片
for _, row in selected_images.iterrows():
    src_path = os.path.join(src_dir, row['image_path'])
    dst_path = os.path.join(dst_dir, row['image_path'])
    
    try:
        shutil.copy2(src_path, dst_path)
        print(f"Copied {row['image_path']} (score: {row['score']:.4f})")
    except Exception as e:
        print(f"Error copying {row['image_path']}: {str(e)}")

print(f"\nCompleted! Copied {len(selected_images)} images")