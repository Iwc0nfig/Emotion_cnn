import torchvision.transforms as transforms
from PIL import Image
import os

# === Paths ===
input_folder  = r"C:\Users\stsa\.cache\kagglehub\datasets\ananthu017\emotion-detection-fer\versions\1\train\disgusted"
output_folder = "train/disgusted"
os.makedirs(output_folder, exist_ok=True)

# === Augmentation config ===
n = 3                
start_index = 0       

# === Transform pipeline (grayscale, 48x48) ===
augment_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.RandomResizedCrop(48, scale=(0.9, 1.0)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,)) I will do that in the training file 
])

to_pil = transforms.ToPILImage(mode='L')  

# === Augmentation loop ===
img_counter = start_index

for filename in os.listdir(input_folder):
    if not filename.lower().endswith('.png'):
        continue  # Skip non-PNG files

    img_path = os.path.join(input_folder, filename)
    original_img = Image.open(img_path).convert("L")  # Ensure grayscale

    for _ in range(n):
        augmented_tensor = augment_transform(original_img)  # (1, 48, 48)
        augmented_pil = to_pil(augmented_tensor)            # Back to grayscale PIL image

        save_path = os.path.join(output_folder, f"im{img_counter}.png")
        augmented_pil.save(save_path)
        img_counter += 1

print(f"✅ Saved {img_counter - start_index} augmented images to '{output_folder}'")
