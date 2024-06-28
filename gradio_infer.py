import gradio as gr
import torch
from torchvision import transforms

from model import ExampleModel

model_path = "model/animal_7.pth"
labels = ["bird", "cat", "dog", "horse"]
num_classes = len(labels)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Preprocess
preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# Load Model
model = ExampleModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# Prediction function
def predict(image):
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(img_tensor)
    prediction = torch.nn.functional.softmax(prediction, dim=1).squeeze()
    confidences = {labels[i]: float(prediction[i]) for i in range(num_classes)}
    return confidences


gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=num_classes),
).launch()