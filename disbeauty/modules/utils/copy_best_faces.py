import pandas as pd
import os
import shutil

# Define paths using relative paths from project root
ROOT_PATH = r"D:\PyProject\Facial Aesthetic Enhancement\Datasets\MEBeauty-database-main"
SCORES_PATH = os.path.join("beauty_scores")  # Relative to DisentangleBeauty
OUTPUT_PATH = os.path.join("Experiment", "Best Faces")  # Up one level then to Experiment

def create_dirs():
    """Create output directory structure"""
    for top_n in ['best-5', 'best-10']:
        for gender in ['female', 'male']:  # Can be expanded for male if needed
            for ethnicity in ['asian', 'black', 'caucasian', 'hispanic', 'indian', 'mideastern']:
                dir_path = os.path.join(OUTPUT_PATH, top_n, gender, ethnicity)
                os.makedirs(dir_path, exist_ok=True)

def process_scores():
    """Process each score file and copy top images"""
    # Get all CSV files
    for gender in ['female', 'male']:  # Can be expanded for male if needed
        gender_path = os.path.join(SCORES_PATH, gender)
        for file in os.listdir(gender_path):
            if file.endswith('.csv'):
                ethnicity = file.split('_')[1].split('.')[0]
                file_path = os.path.join(gender_path, file)
                
                # Read and sort scores
                df = pd.read_csv(file_path)
                df = df.sort_values('score', ascending=False)
                
                # Process top 5 and top 10
                for n, dir_name in [(5, 'best-5'), (10, 'best-10')]:
                    top_n = df.head(n)
                    
                    # Copy each image
                    for _, row in top_n.iterrows():
                        src = os.path.join(ROOT_PATH, row['image_path'])
                        dst = os.path.join(OUTPUT_PATH, dir_name, gender, ethnicity,
                                         os.path.basename(row['image_path']))
                        try:
                            shutil.copy2(src, dst)
                            print(f"Copied {os.path.basename(src)} to {dir_name}/{gender}/{ethnicity}")
                        except Exception as e:
                            print(f"Error copying {src}: {e}")

def main():
    create_dirs()
    process_scores()

if __name__ == "__main__":
    main()