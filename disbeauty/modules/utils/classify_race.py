import os
import shutil
from deepface import DeepFace
from tqdm import tqdm

def setup_directory_structure(base_dir):
    """Create gender/race directory structure"""
    genders = ["male", "female"]
    races = ["asian", "indian", "black", "caucasian", "mideastern", "hispanic"]
    
    for gender in genders:
        gender_path = os.path.join(base_dir, gender)
        os.makedirs(gender_path, exist_ok=True)
        
        for race in races:
            race_path = os.path.join(gender_path, race)
            os.makedirs(race_path, exist_ok=True)

def classify_images(input_dir, output_base_dir):
    """Classify images by gender and race"""
    # Race mapping for consistent naming
    race_mapping = {
        "latino": "hispanic",
        "middle": "mideastern",
        "middle eastern": "mideastern",
        "latino hispanic": "hispanic",
        "white": "caucasian"
    }
    
    # Setup directory structure
    setup_directory_structure(output_base_dir)
    
    # Process each image
    for img_name in tqdm(os.listdir(input_dir)):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        try:
            img_path = os.path.join(input_dir, img_name)
            
            # Analyze face
            result = DeepFace.analyze(img_path,
                                    actions=['gender', 'race'],
                                    enforce_detection=False,
                                    detector_backend='opencv')
            
            # Get gender and race
            gender = 'male' if result[0]['dominant_gender'].lower() == 'man' else 'female'
            dominant_race = max(result[0]['race'].items(), key=lambda x: x[1])[0].lower()
            race = race_mapping.get(dominant_race, dominant_race)
            
            # Construct destination path
            dest_dir = os.path.join(output_base_dir, gender, race)
            dest_path = os.path.join(dest_dir, img_name)
            
            # Move image
            shutil.copy2(img_path, dest_path)
            
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")

if __name__ == "__main__":
    
    input_dir = 'Experiment/Data/ffhq_3000/3000_lowest'
    output_dir = "Experiment/ffhq_3000/3000_lowest_by_gender_race"
    
    if os.path.exists(input_dir):
        classify_images(input_dir, output_dir)
        print(f"Processed folder {input_dir}")

    print("Classification complete!")