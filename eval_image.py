import torch
import argparse
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
input_size=224
model= torch.load("mask_classifier.pth", map_location=torch.device("cpu"))
model.eval()
ap =argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())
image= Image.open(args["image"])
loader= transforms.Compose([ transforms.Resize(input_size),transforms.CenterCrop(input_size),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
image= loader(image).float()
image= Variable(image,requires_grad=True)
image= image.unsqueeze(0)
print(torch.argmax(model(image)))
