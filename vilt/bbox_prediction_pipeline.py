from vilt.modules import ViLTForBBoxPrediction
from vilt.transforms import keys_to_transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from transformers import ViltProcessor

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


model = ViLTForBBoxPrediction.from_pretrained("data/weights/vilt_200k_mlm_itm.ckpt")
transforms = keys_to_transforms(["custom_dataset"], size=384)

def predict_bbox(image_path, text):
    image = Image.open(image_path)
    inputs = processor(image, text, return_tensors="pt")
    outputs = model(**inputs)
    bbox = outputs.logits.detach().numpy()[0]
    
    # Convert normalized coordinates to pixel values
    width, height = image.size
    return [
        bbox[0] * width,
        bbox[1] * height,
        bbox[2] * width,
        bbox[3] * height
    ]



def draw_bbox(image_path, text):
    bbox = predict_bbox(image_path, text)
    image = cv2.imread(image_path)
    
    cv2.rectangle(image, 
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[2]), int(bbox[3])),
        (0,255,0), 2
    )
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
