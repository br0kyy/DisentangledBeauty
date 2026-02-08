import pandas as pd
import os
import shutil
from tqdm import tqdm

# Set paths relative to project root
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(ROOT_DIR, 'Experiment', 'fake.csv')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'Experiment', '1000_lowest')
SOURCE_DIR = os.path.join(ROOT_DIR, 'fake')

# Read CSV file
df = pd.read_csv(CSV_PATH)

# Sort by score ascending
df_sorted = df.sort_values('score', ascending=True)

# Select lowest 1000 scores
lowest_1000 = df_sorted.head(1000)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Created directory: {OUTPUT_DIR}")

# Copy images with progress bar
print(f"\nCopying {len(lowest_1000)} lowest scored images...")
for _, row in tqdm(lowest_1000.iterrows(), total=len(lowest_1000)):
    src = os.path.join(SOURCE_DIR, row['image_path'])
    dst = os.path.join(OUTPUT_DIR, os.path.basename(row['image_path']))
    try:
        shutil.copy2(src, dst)
        print(f"Copied {os.path.basename(row['image_path'])} (score: {row['score']:.3f})")
    except Exception as e:
        print(f"Error copying {row['image_path']}: {e}")

print(f"\nCompleted! Copied {len(lowest_1000)} images to {OUTPUT_DIR}")
print(f"Score range: {lowest_1000['score'].min():.3f} - {lowest_1000['score'].max():.3f}")