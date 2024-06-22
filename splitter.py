import os
import random
import shutil

dataset_path = "./dataset"
train_path = "model_dataset/train"
test_path = "model_dataset/test"
split_ratio = 0.8

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)

    if not os.path.isdir(class_path):
        continue # sonrakine geç

    files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

    random.shuffle(files)

    split_index = int(len(files) * split_ratio)
    train_files = files[:split_index]
    test_files = files[split_index:]

    os.makedirs(os.path.join(train_path, class_folder), exist_ok=True)
    os.makedirs(os.path.join(test_path, class_folder), exist_ok=True)

    for f in train_files:
        shutil.copy2(
            os.path.join(class_path, f), os.path.join(train_path, class_folder)
        )

    for f in test_files:
        shutil.copy2(
            os.path.join(class_path, f), os.path.join(test_path, class_folder)
        )

print("Dataset splitting completed")