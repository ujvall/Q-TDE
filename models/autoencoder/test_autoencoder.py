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

model.load_state_dict(
    torch.load(
        "models/autoencoder/saved/autoencoder.pth",
        map_location=device
    )
)

model.eval()

for label in ["genuine", "tampered"]:
    dataset = QRDataset(f"data/test/{label}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    mean_errors = []
    local_errors = []

    with torch.no_grad():
        for image in loader:
            image = image.to(device)

            reconstructed = model(image)

            error = torch.abs(
                image - reconstructed
            )[0, 0].cpu().numpy()

            mean_errors.append(np.mean(error))

            threshold = np.mean(error) + 2 * np.std(error)
            mask = (error > threshold).astype(np.uint8) * 255

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel
            )

            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)

            regions = []

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]

                if area >= 5:
                    regions.append(area)

            local_errors.append(sum(regions))

    print(label)
    print("Average reconstruction error:", np.mean(mean_errors))
    print("Average spatial anomaly score:", np.mean(local_errors))
