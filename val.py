import os
import shutil
import random

# =========================
# BASE PATH
# =========================
base_path = r"C:\Users\rushi\OneDrive\Desktop\mini_project\disaster_datasets"

# =========================
# FIND ALL IMAGES RECURSIVELY
# =========================
image_paths = []

for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_paths.append(os.path.join(root, file))

# =========================
# CHECK
# =========================
if len(image_paths) == 0:
    print("❌ No images found anywhere!")
    exit()

print(f"✅ Found {len(image_paths)} images")

# =========================
# OUTPUT DATASET
# =========================
base_dir = r"C:\Users\rushi\OneDrive\Desktop\mini_project\dataset"

classes = ['flood', 'fire', 'earthquake', 'cyclone']

for split in ['train', 'val']:
    for cls in classes:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

# =========================
# SPLIT DATA
# =========================
random.shuffle(image_paths)

split_idx = int(0.8 * len(image_paths))
train_imgs = image_paths[:split_idx]
val_imgs = image_paths[split_idx:]

# =========================
# COPY FILES
# =========================
def move_images(image_list, split):
    for img_path in image_list:
        img_name = os.path.basename(img_path)

        # Random class (temporary)
        cls = random.choice(classes)

        dst = os.path.join(base_dir, split, cls, img_name)
        shutil.copy(img_path, dst)

move_images(train_imgs, 'train')
move_images(val_imgs, 'val')

print("✅ Dataset organized successfully!")