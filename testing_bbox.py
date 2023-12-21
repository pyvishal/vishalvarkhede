import torch
import streamlit as st
from PIL import Image, ImageDraw
from pathlib import Path
from torchvision import transforms
from models.experimental import attempt_load
from utils.general import non_max_suppression
from torchvision.transforms import ToPILImage


def load_model(weights_path, device = 'cuda'):
    model = attempt_load(weights_path, map_location=device)
    return model.eval()

# def load_segmentation_model(segment_weights_path, device = 'cuda'):
#     model = attempt_load(segment_weights_path, map_location=device)
#     return model.eval()

def inference_and_save_image(image_path, model, device='cuda', conf_threshold=0.5, iou_threshold=0.45):
    img_size = 640

    #img = torch.from_numpy(transforms.ToTensor()(Path(image_path).read_text()).float())[None]
    img = Image.open(image_path)
    img = transforms.ToTensor()(img).float()
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img)[0]

    prediction = non_max_suppression(prediction, conf_threshold, iou_threshold)[0]

    to_pil = ToPILImage()
    pil_image = to_pil(img[0].cpu())

    for box in prediction:
        if box is not None:
            box = box.cpu().numpy().astype(int)
            pil_image = pil_image.convert('RGB')
            draw = ImageDraw.Draw(pil_image)
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='red', width=2)
    #pil_image.save(save_path)
    return pil_image

def main():
    st.title("Detection App")

    uploaded_image = st.file_uploader("Choose an image....", type = ['jpg','jpeg','png'])
    if uploaded_image is not None:
        weights_path = "./runs/train/yolov7-custom8/weights/best.pt"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        yolov7_model = load_model(weights_path, device=device)

        # segment_weights_path = "./sam_weight/sam_vit_h_4b8939.pth"
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # segment_model = load_model(segment_weights_path, device=device)

        result_image = inference_and_save_image(uploaded_image, yolov7_model,device=device)

        st.image([uploaded_image, result_image], caption=['original_image', 'Detection_output'], width = 300)

if __name__ == '__main__':
    main()

# weights_path = "./runs/train/yolov7-custom8/weights/best.pt"
# image_path = "./inference/spill_image/5.jpg"
# #output_path = "/home/verizon/vision1/Aishwarya2/Aishwarya/yolov7/yolov7-main/test_output/output.jpg"

# yolov7_model = load_model(weights_path)

# images  = inference_and_save_image(image_path, yolov7_model)
# print(type(images))

# output_image_name = 'output_1.jpg'
# output_image_path = '/home/verizon/vision1/Aishwarya2/Aishwarya/yolov7/yolov7-main/test_output/'

# output_image_path = output_image_path + output_image_name
# images.save(output_image_path)
# result = inference_image(image_path, yolov7_model)
# print(type(result))
# print(result)