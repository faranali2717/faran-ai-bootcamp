import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

model = models.resnet18(weights=None) 
model.fc = nn.Linear(model.fc.in_features, 3)

model.load_state_dict(torch.load("clothing_classifier.pth", map_location=torch.device('cpu')))
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classes = ['jeans', 'shoes', 't-shirt']

def predict(image):
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    result = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
    return result

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Clothing AI (Retail Startup MVP)",
    description="Upload an image of a T-shirt, Jeans, or Shoes. The AI will classify it."
)

if __name__ == "__main__":
    interface.launch()
