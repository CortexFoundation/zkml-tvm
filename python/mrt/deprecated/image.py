from os import path

import numpy as np
from PIL import Image

import tvm
from tvm.contrib.download import download_testdata


def get_real_image(im_height, im_width) -> np.ndarray:
    repo_base = "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/"
    img_name = "elephant-299.jpg"
    image_url = path.join(repo_base, img_name)
    img_path = download_testdata(image_url, img_name, module="data")
    image = Image.open(img_path).resize((im_height, im_width))
    data = np.array(image).astype("float32")
    data = np.reshape(data, (1, im_height, im_width, 3))
    data = np.transpose(data, (0, 3, 1, 2))
    return data
