import torch
from PIL import Image 

from model import Model
from torchvision import transforms


def Predictor():
    extractor = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=True).features
    model = Model(extractor)
    for param in model.parameters():
        param.requires_grad = False
    checkpoint = torch.load('model.pth.tar',map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])

    def predictor(image):

        image_transform =  transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        ])

        pred = model(torch.unsqueeze(image_transform(image), 0))[0]
        gender_pred= float(pred[0])
        age_pred = float(pred[1])

        age_pred *= 100.0
        if gender_pred < 0.5:
            gender_pred = 0.0
        else:
            gender_pred = 1.0
    
        return age_pred,gender_pred
    return predictor
