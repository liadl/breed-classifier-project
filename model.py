import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# Check if GPU is available to speed up training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom classification head to replace the default ResNet layer
class CustomCnnHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) # Dropout to avoid overfitting
        self.fc2 = nn.Linear(1024, num_classes)    
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x      


def train_model():
    # Data augmentation to help the model generalize better
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
    ]) 

    # No need for augmentation on the test set
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
    ])

    # Load the pets dataset
    train_dataset = datasets.OxfordIIITPet(root="./dataset", split="trainval", target_types="category", download=True, transform=train_transform)
    test_dataset = datasets.OxfordIIITPet(root="./dataset", split="test", target_types="category", download=True, transform=test_transform)
            
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    num_classes = len(train_dataset.classes)
    
    # Use pre-trained ResNet50 so we don't start from scratch
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze base layers to keep pre-trained features
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last layer with our custom head
    features_size = model.fc.in_features
    model.fc = CustomCnnHead(in_features=features_size, num_classes=num_classes)
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Training loop (5 epochs)
    for epoch in range(5):
        model.train() # Training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0
                
        for X_train, Y_train in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            X_train, Y_train = X_train.to(device), Y_train.to(device)
                    
            optimizer.zero_grad() # Clear old gradients
            outputs = model(X_train)
            loss = criterion(outputs, Y_train)
            loss.backward() # Backprop
            optimizer.step() # Update weights
                
            running_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            correct_train += (predictions == Y_train).sum().item()
            total_train += Y_train.size(0)
                   
        train_loss = running_loss / len(train_loader)
        train_acc = (correct_train / total_train) * 100
                
        # Testing phase
        model.eval() # Evaluation mode
        test_loss = 0.0
        correct_test = 0
        total_test = 0
                
        with torch.no_grad(): # No need to track gradients here
            for X_test, Y_test in tqdm(test_loader, desc=f"Test  Epoch {epoch+1}"):
                X_test, Y_test = X_test.to(device), Y_test.to(device)
                        
                outputs_test = model(X_test)
                loss_test = criterion(outputs_test, Y_test)
                        
                test_loss += loss_test.item()
                _, prediction_test = torch.max(outputs_test, 1)
                correct_test += (prediction_test == Y_test).sum().item()
                total_test += Y_test.size(0)
                        
        final_test_loss = test_loss / len(test_loader)
        test_acc = (correct_test / total_test) * 100

        print(f"\n[Summary] Epoch {epoch + 1}:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss:  {final_test_loss:.4f} | Test Acc:   {test_acc:.2f}%\n")

    # Save the model
    torch.save(model.state_dict(), 'resnet50_pets_augmented.pth')

if __name__ == '__main__':
    train_model()