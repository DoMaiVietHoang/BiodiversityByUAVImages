import os
train_dir = "./Train"
test_dir = "./Test"

train_image_counts = {}
test_image_counts = {}

for class_folder in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_folder)
    if os.path.isdir(class_path):
        num_images = len(os.listdir(class_path))
        train_image_counts[class_folder] = num_images

for class_folder in os.listdir(test_dir):
    class_path = os.path.join(test_dir, class_folder)
    if os.path.isdir(class_path):
        num_images = len(os.listdir(class_path))
        test_image_counts[class_folder] = num_images


print("Số lượng ảnh trong mỗi lớp của tập huấn luyện:")
for class_folder, count in train_image_counts.items():
    print(f"{class_folder}: {count} ảnh")

print("\nSố lượng ảnh trong mỗi lớp của tập kiểm tra:")
for class_folder, count in test_image_counts.items():
    print(f"{class_folder}: {count} ảnh")
