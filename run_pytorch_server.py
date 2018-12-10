
import io
import json
import numpy as np

import flask
from flask_cors import CORS
import os
import glob

import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torchvision import transforms 
import libs.model_utils as model_utils
import libs.plot_utils as plot_utils
from libs.custom_layers import Flatten


# Initialize our Flask application and the PyTorch model.
app = flask.Flask(__name__)
CORS(app)
model = None

def load_model():
    """Load the pre-trained model, you can use your model just as easily.

    """
    global model
    custom_head = nn.Sequential(
        Flatten(),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512*7*7, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),
        nn.Linear(256, 4+20)
    )

    model = model_utils.get_resnet34_model_with_custom_head(custom_head)
    model.load_state_dict(torch.load('combined_model_val_77.5.ckpt', map_location='cpu'))

    model.eval()

def get_category_to_label(id):
    id_to_cat = {
        0: 'car',
        1: 'horse',
        2: 'person',
        3: 'aeroplane',
        4: 'train',
        5: 'dog',
        6: 'chair',
        7: 'boat',
        8: 'bird',
        9: 'pottedplant',
        10: 'cat',
        11: 'sofa',
        12: 'motorbike',
        13: 'tvmonitor',
        14: 'bus',
        15: 'sheep',
        16: 'diningtable',
        17: 'bottle',
        18: 'cow',
        19: 'bicycle'}
    return id_to_cat[id]

def test_model_on_img(im):
    sz = 224
    test_tfms = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor()
    ])
    test_im_tensor = test_tfms(im)[None]
    
    pred_bbox, pred_cat_id, conf = model_utils.test_on_single_image(test_im_tensor, model, sz)
    return plot_utils.get_result_on_test_image(pred_bbox, pred_cat_id, conf, get_category_to_label, im)

def delete_other_result_imgs(folder):
    files = glob.glob('app/results/'+folder+'/*.png')
    if(len(files)>=10):
        for file in files:
            os.remove(file)
    return


def apply_sobel_operator_on_custom_image(img):
    T = transforms.ToTensor()
    P = transforms.ToPILImage()
    img_bw = img.convert('L')
    x = T(img_bw)[None]
    
    #Black and white input image x, 1x1xHxW
    a = torch.Tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])
        
    a = a.view((1,1,3,3))
    G_x = F.conv2d(x, a, padding=1)
      
    b = torch.Tensor([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]])
      
    b = b.view((1,1,3,3))
    G_y = F.conv2d(x, b, padding=1)
      
    G = torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
    im = P(G[0])
    
    hash = np.random.randint(low=0, high=9, size=10)
    hash = ''.join(str(i) for i in hash)
    res_url = 'results/sobel-operator/res-'+hash+'.png'
    im.save('app/'+res_url)
    return res_url


@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            # Read the image in PIL format
            delete_other_result_imgs('largest-item-bbox')

            image_str = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image_str))
            res_url = test_model_on_img(image)
            data['res_url'] = res_url

            # Indicate that the request was a success.
            data["success"] = True
    return flask.jsonify(data)    

@app.route("/detect_edge", methods=["POST"])
def detect_edge():
    data = {"success": False}
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            delete_other_result_imgs('sobel-operator')
            image_str = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image_str))
            res_url = apply_sobel_operator_on_custom_image(image)
            data['res_url'] = res_url
            
            # Indicate that the request was a success.
            data["success"] = True
    return flask.jsonify(data)

if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    load_model()
    app.run(host='0.0.0.0', port=5000)

#return send_file(output,
#                     attachment_filename='logo.png',
#                     mimetype='image/png')
