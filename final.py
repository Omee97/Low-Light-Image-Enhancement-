import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

# ==============================
# 1️⃣ HTFNet Model Definition
# ==============================

# Histogram Feature Extractor
class HistogramFeatureExtractor(nn.Module):
    def __init__(self, bins=128):
        super(HistogramFeatureExtractor, self).__init__()
        self.bins = bins
        self.fc = nn.Linear(bins * 3, 128)  # Expects input of size [batch_size, bins*3]

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        all_features = []
        
        for b in range(batch_size):
            sample_features = []
            for c in range(channels):
                # Create histogram for each channel of each sample
                hist = torch.histc(x[b, c, :, :].flatten(), bins=self.bins, min=0, max=1)
                hist = hist / (height * width)  # Normalize
                sample_features.append(hist)
            
            # Concatenate the histograms of all channels for this sample
            sample_features = torch.cat(sample_features)  # Shape: [bins*3]
            all_features.append(sample_features)
        
        # Stack all samples
        all_features = torch.stack(all_features)  # Shape: [batch_size, bins*3]
        
        # Apply fully connected layer
        output_features = F.relu(self.fc(all_features))  # Shape: [batch_size, 128]
        
        return output_features


# Spatial Feature Extractor
class SpatialFeatureExtractor(nn.Module):
    def __init__(self):
        super(SpatialFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv(x)

# Transformation Function Estimator
class TransformationFunctionEstimator(nn.Module):
    def __init__(self):
        super(TransformationFunctionEstimator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(128 + 128, 128),  # 128 from spatial and 128 from histogram
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
    
    def forward(self, spatial_features, hist_features):
        # Global average pooling on spatial features to get [batch_size, 128]
        spatial_features = spatial_features.mean(dim=[2, 3])
        
        
        # Both should now be [batch_size, 128]
        combined_features = torch.cat((spatial_features, hist_features), dim=1)
        return self.fc(combined_features)


# HTFNet Model
class HTFNet(nn.Module):
    def __init__(self):
        super(HTFNet, self).__init__()
        self.hist_extractor = HistogramFeatureExtractor()
        self.spatial_extractor = SpatialFeatureExtractor()
        self.transformation_estimator = TransformationFunctionEstimator()
    
    def forward(self, x):
        hist_features = self.hist_extractor(x)  # [batch_size, 128]
        spatial_features = self.spatial_extractor(x)  # [batch_size, 128, H, W]
        transform_params = self.transformation_estimator(spatial_features, hist_features)
        enhanced = x * transform_params.view(-1, 3, 1, 1)
        return torch.clamp(enhanced, 0, 1)


# ==============================
# 2️⃣ Dataset Loader
# ==============================

class LowLightDataset(Dataset):
    def __init__(self, lowlight_dir, enhanced_dir, transform=None):
        self.lowlight_dir = lowlight_dir
        self.enhanced_dir = enhanced_dir
        self.lowlight_images = sorted(os.listdir(lowlight_dir))
        self.enhanced_images = sorted(os.listdir(enhanced_dir))
        self.transform = transform

    def __len__(self):
        return len(self.lowlight_images)

    def __getitem__(self, idx):
        lowlight_path = os.path.join(self.lowlight_dir, self.lowlight_images[idx])
        enhanced_path = os.path.join(self.enhanced_dir, self.enhanced_images[idx])
        
        lowlight = Image.open(lowlight_path).convert("RGB")
        enhanced = Image.open(enhanced_path).convert("RGB")

        if self.transform:
            lowlight = self.transform(lowlight)
            enhanced = self.transform(enhanced)

        return lowlight, enhanced


# ==============================
# 3️⃣ Training Function
# ==============================

def train_model(model, dataloader, num_epochs=20, learning_rate=0.0002, model_path="htfnet.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # Loss function

    for epoch in range(num_epochs):
        epoch_loss = 0
        for lowlight, enhanced in dataloader:
            lowlight, enhanced = lowlight.to(device), enhanced.to(device)
            
            optimizer.zero_grad()
            output = model(lowlight)
            loss = criterion(output, enhanced)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader)}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


# ==============================
# 4️⃣ Image Enhancement (for testing)
# ==============================

def enhance_image(model, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        enhanced_image = model(image)
    
    enhanced_image = enhanced_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    enhanced_image = (enhanced_image * 255).astype(np.uint8)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(Image.open(image_path))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Enhanced")
    plt.imshow(enhanced_image)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save the enhanced image
    enhanced_path = image_path.replace('.jpg', '_enhanced.jpg')
    Image.fromarray(enhanced_image).save(enhanced_path)
    print(f"Enhanced image saved to {enhanced_path}")

# ==============================
# 5️⃣ Running the Code
# ==============================

if __name__ == "__main__":
    lowlight_dir = r"E:\Image Processing\project\LOLdataset\our485\low"
    enhanced_dir = r"E:\Image Processing\project\LOLdataset\our485\high"
    
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset = LowLightDataset(lowlight_dir, enhanced_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = HTFNet()
    train_model(model, dataloader)
    
    enhance_image(model, r"E:\Image Processing\project\WhatsApp Image 2025-03-28 at 11.49.19_e19a98e0.jpg")