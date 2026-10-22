import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = ViTImageProcessor.from_pretrained("models/vit/saved")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=processor.image_mean,
        std=processor.image_std
    )
])

test_data = ImageFolder("data/test", transform=transform)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

model = ViTForImageClassification.from_pretrained("models/vit/saved")
model.to(device)
model.eval()

actual = []
predicted = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(pixel_values=images)
        predictions = torch.argmax(outputs.logits, dim=1)

        actual.extend(labels.numpy())
        predicted.extend(predictions.cpu().numpy())

print("Test Accuracy:", accuracy_score(actual, predicted))
print("Precision:", precision_score(actual, predicted, zero_division=0))
print("Recall:", recall_score(actual, predicted, zero_division=0))
print("F1 Score:", f1_score(actual, predicted, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(actual, predicted))
