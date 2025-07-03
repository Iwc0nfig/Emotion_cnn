import os 
from torch.utils.data import Dataset, DataLoader 
from PIL import Image
import torch
import  torchvision.transforms  as transforms
from tqdm import tqdm
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json

train_path = "train/"
val_path = 'val/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

loss_dict = {}


class EmotionDataset(Dataset):
    def __init__(self,folder_path,transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images = []
        self.labels = []

        emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

        for label , emotion in enumerate(emotions):
            emotion_folder = os.path.join(self.folder_path,emotion)
            if os.path.exists(emotion_folder):
                for img_file in os.listdir(emotion_folder):
                    img_path = os.path.join(emotion_folder,img_file)
                    self.images.append(img_path)
                    self.labels.append(label)
            else:
                continue
        print(f"Loaded {len(self.images)} images from {folder_path}")

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path).convert('L') 
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        return image,torch.tensor(label,dtype=torch.long)
    

def create_dataloaders_train(train_folder, batch_size=64):
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((48,48)),
        transforms.Grayscale(num_output_channels=1), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomErasing(p=0.35, scale=(0.02, 0.08), ratio=(0.3, 3.3))

    ])

    


    train_dataset = EmotionDataset(train_folder, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True ,pin_memory=True )
    return train_loader ,train_dataset

def create_dataloaders_val(val_path,batch_size=64):
    val_transform = transforms.Compose([
        transforms.Resize((48,48)),
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    val_dataset = EmotionDataset(val_path, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False ,pin_memory=True)
    return val_loader

train_loader , train_dataset  =create_dataloaders_train(train_path,batch_size=64)
val_loader = create_dataloaders_val(val_path,batch_size=64)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(

            nn.Conv2d(1,32,3, padding=1), nn.BatchNorm2d(32),nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Dropout(0.25),

            


        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.net(x)
        return self.classifier(x)



#These are my old setting 
"""
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

"""



model = Net()
model.to(device)
#hyparammeters 
epochs =60   
learning_rate = 1e-3 

optimizer = torch.optim.AdamW(model.parameters() , lr=learning_rate , weight_decay=5e-4)
#add balance
train_labels = [label.item() for _, label in train_dataset]
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float().to(device), label_smoothing=0.1)

#I add scheduler to the learning rate .This will change the lr dynamicly 
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=3, factor=0.5)


#not necesary but it print the size of the model
size = sum([p.numel() for p in model.parameters()])
print(size)

def model_training():
    print("Start training >>> ")
    

    
    
    for epoch in range(epochs):
        # Training phase
        model.train()  # Set to training mode
        train_loss = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()  # Set to evaluation mode
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate average validation loss and accuracy
        avg_val_loss = val_loss /total
        val_accuracy = 100 * correct / total
        
        # Step the scheduler with validation loss
        scheduler.step(avg_val_loss)
        model.train()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        loss_dict[epoch] = {"Train Loss": avg_train_loss , "Val Loss": avg_val_loss}
        
        print(f"Epoch: {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | LR: {current_lr:.6f}")

def save_model(full_model = False):
    if full_model:
        torch.save(model, "full_model.pth")
        print("Entire model saved successfully!") #when you load it you will need to evaluate the model (model.eval())
    else:
        torch.save(model.state_dict(), "model.pth") #To use it you will need ot have the class Net in your code 
        print("Model saved successfully!")



def main():
    model_training()
    print(loss_dict)
    save_model()

    with open('output.txt', 'w') as f:
        json.dump(loss_dict,f,indent=4)

if __name__ == "__main__":
    main()