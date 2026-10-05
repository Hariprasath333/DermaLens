import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import csv

class LesionIQDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = []
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)
                
        self.transform = transform
        
        self.meta_cols = [
            'age_approx', 'sex_female', 'sex_male', 'sex_unknown',
            'site_anterior torso', 'site_head/neck', 'site_lateral torso',
            'site_lower extremity', 'site_oral/genital', 'site_palms/soles',
            'site_posterior torso', 'site_unknown', 'site_upper extremity'
        ]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data[idx]
        
        # Load image
        img_path = row['image_path']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # Metadata tensor
        meta_values = []
        for col in self.meta_cols:
            val = row[col]
            if val == '' or val is None:
                meta_values.append(0.0)
            elif val in ['True', 'False', True, False]:
                meta_values.append(1.0 if str(val) == 'True' else 0.0)
            else:
                meta_values.append(float(val))
                
        # Normalize age_approx (divide by 100 to keep it roughly 0-1)
        meta_values[0] = meta_values[0] / 100.0
        
        meta_tensor = torch.tensor(meta_values, dtype=torch.float32)
        
        # Label
        label = int(float(row['class_encoded']))
        
        return image, meta_tensor, label

if __name__ == "__main__":
    # Sanity Check
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])
    
    csv_path = r"path/to/layer0_train.csv"
    print(f"Loading dataset from {csv_path}...")
    dataset = LesionIQDataset(csv_path, transform=transform)
    print(f"Dataset length: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print("Fetching one batch...")
    images, meta_tensors, labels = next(iter(dataloader))
    
    print("\n✅ Sanity Check Passed!")
    print(f"Images shape: {images.shape}")
    print(f"Meta tensor shape: {meta_tensors.shape} (meta_dim = {meta_tensors.shape[1]})")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample metadata (first item): {meta_tensors[0]}")
