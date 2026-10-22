import os
import cv2
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QRDataset(Dataset):
    def __init__(self, root):
        self.files = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.endswith(".png")
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = cv2.imread(self.files[index], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 128))

        blur = cv2.GaussianBlur(image, (5, 5), 0)
        detail = cv2.absdiff(image, blur)
        detail = detail.astype(np.float32) / 255.0

        return torch.tensor(detail, dtype=torch.float32).unsqueeze(0)

dataset = QRDataset("data/train/genuine")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.MSELoss()

for epoch in range(15):
    model.train()
    total_loss = 0

    for images in loader:
        images = images.to(device)

        reconstructed = model(images)
        loss = criterion(reconstructed, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/15 - "
        f"Loss: {total_loss / len(loader):.6f}"
    )

os.makedirs("models/autoencoder/saved", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/autoencoder/saved/autoencoder.pth"
)

print("Detail autoencoder saved successfully")
print("Device:", device)
