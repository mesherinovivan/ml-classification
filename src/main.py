import io

import torch
import torchvision
from PIL import Image
from fastapi import FastAPI, File
from fastapi import FastAPI, File, Form, UploadFile
from torchvision import transforms

app = FastAPI()

# load a pre-trained Model and convert it to eval mode.
# This model loads just once when we start the API.
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
model.load_state_dict(torch.load('models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'))
model.eval()

def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    return image


@app.put("/predict")
async def get_predictionbase64(file: UploadFile = File(...), threshold: float = Form(...)):
    '''
	FastAPI API will take a base 64 image as input and return a json object
	'''
    # Load the image
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    img = read_imagefile(await file.read())

    # Convert image to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    # get prediction on image
    pred = model([img])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(float(i[0]), float(i[1])), (float(i[2]), float(i[3]))] for i in
                  list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return {'boxes': pred_boxes,
            'classes': pred_class }
