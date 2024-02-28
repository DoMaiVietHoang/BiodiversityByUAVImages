import os
import shutil
from sklearn.model_selection import train_test_split
import glob
data_dir = "./Data"
train_dir = "./Train"
test_dir = "./Test"
train_ratio = 0.8
for class_folder in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_folder)
    if os.path.isdir(class_path):
        os.makedirs(os.path.join(train_dir, class_folder), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_folder), exist_ok=True)
        image_files = glob.glob(os.path.join(class_path, "*.jpg"))  
        train_images, test_images = train_test_split(image_files, train_size=train_ratio, random_state=42)
        for image_file in train_images:
            shutil.copy(image_file, os.path.join(train_dir, class_folder))
        for image_file in test_images:
            shutil.copy(image_file, os.path.join(test_dir, class_folder))