from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from typing import List
from pathlib import Path
import torch.nn as nn
import torch.optim as optim


NUM_CLASSES = 29
LABELS = [chr(i) for i in range(65, 91)] + ["SPACE", "DELETE", "NOTHING"]

class ConvNet(nn.Module):
    def __init__(self, num_classes=29):
        super(ConvNet, self).__init__()
        
        # 1st convolution layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 2nd convolution layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 3rd convolution layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.bn1(x)
        
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.bn2(x)
        
        x = self.pool3(torch.relu(self.conv3(x)))
        x = self.bn3(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
# Initialize FastAPI app
app = FastAPI()

project_dir = Path(__file__).resolve().parent.parent
model_path = project_dir / "model" / "asl_model.pth"

print(model_path)
# model_path = "/app/models/asl.model.pth"
if not model_path.is_file():
    raise FileNotFoundError(f"Model file not found at: {model_path}")


model = ConvNet(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(model_path))

model.eval()  # Set the model to evaluation mode

# Request body structure
class ImageRequest(BaseModel):
    image: List[float]  # The flattened 28x28 image data

@app.post("/predict")
async def predict(image_request: ImageRequest):
    # Reshape the image data to match the model input (1, 32, 32)
    image = np.array(image_request.image).reshape(1, 1, 32, 32)
    image_tensor = torch.tensor(image, dtype=torch.float32)

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).item()
        prediction_label = LABELS[prediction]

    return {"prediction": prediction_label}