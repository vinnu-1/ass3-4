import streamlit as st
import urllib.parse as parse
import os
import requests
import torch
from PIL import Image
from transformers import *
from tqdm import tqdm
from transformers import ViTImageProcessor,ViTForImageClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

# the model name
model_name = "google/vit-base-patch16-224"
# load the image processor
image_processor = ViTImageProcessor.from_pretrained(model_name)

st.title('DEEP FAKE IMAGE CLASSIFICATION')

st.sidebar.header('CONTROL MENU')

if st.sidebar.button('ABOUT'):
    st.header("PROBLEM STATEMENT")
    st.markdown("Deepfake Al is a type of artificial intelligence used to create convincing images, audio and video hoaxes. The term describes both the technology and the resulting bogus content and is a portmanteau of deep learning and fake. Deepfakes often transform existing source content where one person is swapped for another. They also create entirely original content where someone is represented doing or saying something they didn't do or say. The greatest danger posed by deepfakes is their ability to spread false information that appears to come from trusted sources.")
    st.header("DATASET")
    st.markdown("""This dataset contains manipulated images and real images. The manipulated images are the faces which are created by various means. The source for this dataset was https://zenodo.org/record/5528418#.YpdlS2hBzDd
this dataset was processed as our will to get maximum outcome out of these images. Each image is a 256 X 256 jpg image of human face either real or fake""")

if st.sidebar.button('EXPLORATORY DATA ANALYSIS'):
    pass

if st.sidebar.button('CNN Vs VIT'):
    pass

if st.sidebar.button('FINE TUNING VIT'):
    pass

if st.sidebar.button('EVALUATION'):
    pass


# load the best model, change the checkpoint number to the best checkpoint
# if the last checkpoint is the best, then ignore this cell
best_checkpoint = 2000
# best_checkpoint = 150
model = ViTForImageClassification.from_pretrained(f"./vit-base-deepfake/checkpoint-{best_checkpoint}").to(device)
# model = ViTForImageClassification.from_pretrained(f"./vit-base-skin-cancer/checkpoint-{best_checkpoint}").to(device)


# a function to determine whether a string is a URL or not
def is_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False
    
# a function to load an image
def load_image(image_path):
    if is_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)


def get_prediction(model, url_or_path):
  # load the image
  img = load_image(url_or_path)
  # preprocessing the image
  pixel_values = image_processor(img, return_tensors="pt")["pixel_values"].to(device)
  # perform inference
  output = model(pixel_values)
  # get the label id and return the class name
  return model.config.id2label[int(output.logits.softmax(dim=1).argmax())]

source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
if st.sidebar.button("INFERENCE"):
    if source_img is not None:
            uploaded_image = Image.open(source_img)
            #st.image(source_img, caption="Uploaded Image",use_column_width=True)
            uploaded_image.save('temp.jpg')
            st.image('temp.jpg')
            res = get_prediction(model, "C:/PROJECT/temp.jpg")
            #st.write(res)
            st.markdown("<h2 style='text-align: center; color: #ff5733;'>Prediction Result:</h2>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center; color: #ff5733;'><i class='material-icons'>Image is </i> {}</h3>".format(res), unsafe_allow_html=True)
            

if st.sidebar.button('OUR TEAM'):
    pass



