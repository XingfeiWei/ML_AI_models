import torch
from torchvision import datasets, transforms, models
from PIL import Image

import os
import zipfile
from pathlib import Path
import random
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Check if CUDA is available, and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the AlexNet model structure
alexnet = models.alexnet(pretrained=True)

# Modify the first convolution layer to accept 1 channel instead of 3
alexnet.features[0] = torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)

# Modify the final layer if you changed it during your initial training
num_classes = 6  # Replace with your number of classes
alexnet.classifier[6] = torch.nn.Linear(alexnet.classifier[6].in_features, num_classes)

# Load your custom trained weights
alexnet.load_state_dict(torch.load('./alexnet_trained.pth'))
alexnet.eval()  # Set the model to evaluation mode

# Assuming `model` is your loaded and prepared model

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),          # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean and std
                         std=[0.229, 0.224, 0.225])
])

# images folder directory to be predicted
prediction_folder = '../../predictVMD'

# Replace this with your actual mapping
class_names = {
    0: '1QD-1origami',
    1: '1QD-2origami',
    2: '1QD-3origami',
    3: '1QD-4origami',
    4: '1QD-5origami',
    5: '1QD-6origami'
}

# Function to predict a single image
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    #image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    output = alexnet(image)
    probability = torch.nn.functional.softmax(output, dim=1)[0] * 100
    _, predicted_class = torch.max(output, 1)
    return class_names[predicted_class.item()], probability.tolist()

# Predict and save results
with open('alexnet_trained_predictions_vmd_v4.csv', 'w') as f:
    f.write("Image Name, Predicted Class, Probabilities\n")
    # Get all image files and sort them
    image_files = [file for file in os.listdir(prediction_folder) if file.endswith(('jpg', 'png', 'jpeg'))]
    image_files.sort()  # Sort the list of files
    for image_file in image_files:
        if image_file.endswith(('jpg', 'png', 'jpeg')):  # Check image file extension
            image_path = os.path.join(prediction_folder, image_file)
            predicted_class, probabilities = predict_image(image_path)
            probs_str = ', '.join([f"{p:.2f}" for p in probabilities])
            f.write(f"{image_file}, {predicted_class}, {probs_str}\n")
