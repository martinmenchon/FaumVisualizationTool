import streamlit as st
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageColor
import seaborn as sns
import SessionState
import tempfile
import streamlit.components.v1 as components
session_state = SessionState.get(loaded=False,day=[],upload_key = None,values_to_keep=[], trasnparency=-1)
import base64
import pathlib
import copy

# HACK This only works when we've installed streamlit with pipenv, so the
# permissions during install are the same as the running process
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
# We create a downloads directory within the streamlit static asset directory
# and we write output files to it
DOWNLOADS_PATH = (STREAMLIT_STATIC_PATH / "downloads")
if not DOWNLOADS_PATH.is_dir():
    DOWNLOADS_PATH.mkdir()


# Paths
image_path = "imagen.jpg"
raster_path = "raster.pgm"
output_path = "result.png"

st.set_page_config(
    page_title="FAUM'S VISUALIZATION TOOL",
    page_icon="favicon.png",
    layout="wide",
    #initial_sidebar_state="expanded",
)

# add_selectbox = st.sidebar.selectbox(
#     "How would you like to be contacted?",
#     ("Email", "Home phone", "Mobile phone")
# )

from PIL import Image
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image


#Remove button
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#Containers
header = st.beta_container()
body = st.beta_container()
body1 = st.beta_container()
body2 = st.beta_container()


def build_color_pallete(array_of_values):
    font = cv2.FONT_HERSHEY_SIMPLEX
    SIZE = 30
    color_pallete = np.ones((SIZE,SIZE,3), np.uint8)
    color_pallete[:,:,0] = colors[0][0]
    color_pallete[:,:,1] = colors[0][1]
    color_pallete[:,:,2] = colors[0][2]
    number = np.ones((SIZE,SIZE,3), np.uint8)
    number *= 255
    cv2.putText(number,str(array_of_values[0]),(4,15), font,0.45,(0,0,0),2)
    color_pallete = cv2.vconcat([color_pallete, number])
    for i in (range(1,len(array_of_values))):
        another_color = np.ones((SIZE,SIZE,3), np.uint8)
        another_color[:,:,0] = colors[i][0]
        another_color[:,:,1] = colors[i][1]
        another_color[:,:,2] = colors[i][2]
        number = np.ones((SIZE,SIZE,3), np.uint8)
        number *= 255
        cv2.putText(number,str(array_of_values[i]),(4,15), font,0.45,(0,0,0),2)
        another_color = cv2.vconcat([another_color, number])
        color_pallete = cv2.hconcat([color_pallete, another_color])

    return color_pallete

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def colorize_raster(dim,raster,array_of_values):
    colorized_raster = np.zeros(dim)
    colorized_raster = colorized_raster.astype(np.uint8)
    for value in array_of_values:
        mask = np.isin(raster, value)
        mask = mask*1
        mask = mask.astype(np.uint8)
        r_mask = mask * colors[value-1][0] #red
        g_mask = mask * colors[value-1][1] #green
        b_mask = mask * colors[value-1][2] #blue
        colorized_raster[:, :, 0] += r_mask
        colorized_raster[:, :, 1] += g_mask
        colorized_raster[:, :, 2] += b_mask
    return colorized_raster


with header:
    st.title("FAUM'S VISUALIZATION TOOL")
    
with body1:   
    # Uploading the File to the Page
    uploadFile = st.file_uploader(label="Upload image")

    # Checking the Format of the page
    if uploadFile is not None:
        imagen = load_image(uploadFile)
        st.write("Image Uploaded Successfully")
    # else:
    #     st.write("Make sure you image is in JPG/PNG Format.")

        if session_state.loaded == False:
            raster = cv2.imread(raster_path,-1)

            resized_image = image_resize(imagen, width = 725)

            array_of_values = [] #podria usar range
            for i in range(1,raster.max()+1):
                array_of_values.append(i)

            # build color list
            color_list = sns.color_palette("hls", len(array_of_values)).as_hex()
            colors = []
            for color in color_list:
                rgb_color = ImageColor.getrgb(color)
                colors.append(rgb_color)

            colorized_raster = colorize_raster(np.shape(imagen),raster,array_of_values)
            resized_colorized_raster = image_resize(colorized_raster, width = 725)
            resized_image = cv2.hconcat([resized_image, resized_colorized_raster])

            color_pallete = build_color_pallete(array_of_values)

            # Save state
            session_state.array_of_values = array_of_values
            session_state.imagen = imagen
            session_state.raster = raster
            session_state.colorized_raster = colorized_raster
            session_state.colors = colors
            session_state.resized_image = resized_image 
            session_state.color_pallete = color_pallete
            session_state.loaded = True

        else:
            array_of_values = session_state.array_of_values
            imagen = session_state.imagen
            raster = session_state.raster
            colorized_raster = session_state.colorized_raster
            colors = session_state.colors
            resized_image = session_state.resized_image
            color_pallete = session_state.color_pallete

        st.image(session_state.resized_image)
        st.image(color_pallete) 

if uploadFile is not None:       
    with body2:
        values_to_keep = st.multiselect("Select values you want to keep", array_of_values)
        trasnparency = st.slider('Select a transparency',value=128, min_value=0, max_value=255)
        if session_state.values_to_keep != values_to_keep or session_state.trasnparency != trasnparency:
            download_buton_pressed = False
        else:
            download_buton_pressed = True

        if st.button('Filter image') or download_buton_pressed:
            session_state.values_to_keep = values_to_keep
            session_state.trasnparency = trasnparency
            st.title("Image filtered")
            copied_image = copy.copy(imagen)
            copied_image = cv2.cvtColor(copied_image, cv2.COLOR_RGB2BGR)
            bgra = cv2.cvtColor(copied_image, cv2.COLOR_BGR2BGRA)
            # Then assign the mask to the last channel of the image
            bgra[:, :, 3] = trasnparency

            mask = np.isin(raster, values_to_keep)
            m = mask*1
            m = m*(255-trasnparency)
            m = m.astype(np.uint8)
            bgra[:, :, 3] += m

            cv2.imwrite(str(DOWNLOADS_PATH / "result.png"), bgra)

            out = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
            out_resized = image_resize(out, width = 1000)
            st.image(out_resized)
            st.markdown("Download filered image from [here](downloads/result.png)")