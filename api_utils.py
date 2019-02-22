import io
import json
import numpy as np

import os
import glob

import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torchvision import transforms 
import libs.model_utils as model_utils
import libs.plot_utils as plot_utils
from libs.custom_layers import Flatten, AdaptiveConcatPool
from libs.custom_transforms import Resize


def load_single_obj_det_model():
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
    return model

def load_multi_class_model():
    custom_head = nn.Sequential(
        AdaptiveConcatPool(),
        Flatten(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.25),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 20),
        nn.Sigmoid()
    )
    model = model_utils.get_resnet34_model_with_custom_head(custom_head)
    model.load_state_dict(torch.load('multi_class.ckpt', map_location='cpu'))
    model.eval()
    return model

def load_yoga_pose_classifier_model():
    model = model_utils.get_resnet34_model()
    model.fc = nn.Linear(512, 107)  
    model.load_state_dict(torch.load('yoga-asana-classifier.ckpt', map_location='cpu'))
    model.eval()
    return model

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


def test_model_on_img(im, model):
    sz = 224
    test_tfms = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor()
    ])
    test_im_tensor = test_tfms(im)[None]
    
    pred_bbox, pred_cat_id, conf = model_utils.test_on_single_image(test_im_tensor, model, sz)
    return plot_utils.get_result_on_test_image(pred_bbox, pred_cat_id, conf, get_category_to_label, im)

def get_multi_class_labeled_image(im, model):
    sz = 224
    test_tfms = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor()
    ])
    test_im_tensor = test_tfms(im)[None]
    
    pred_classes, pred_probs = model_utils.get_multi_class_labeled_image(test_im_tensor, model)
    return plot_utils.get_multi_class_labeled_image(pred_classes, pred_probs, im)

def get_yoga_pose_labeled_image(im, model):
    pose_id_to_name = {0: 'Bharadvajasana I', 1: 'Padangusthasana', 2: 'Paripurna Navasana', 3: 'Baddha Konasana', 4: 'Dhanurasana', 5: 'Setu Bandha Sarvangasana', 6: 'Ustrasana', 7: 'Marjaryasana', 8: 'Chakravakasana', 9: 'Ashtanga Namaskara', 10: 'Utkatasana', 11: 'Balasana', 12: 'Bhujangasana', 13: 'Savasana', 14: 'Gomukhasana', 15: 'Bitilasana', 16: 'Bakasana', 17: 'Makara Adho Mukha Svanasana', 18: 'Ardha Pincha Mayurasana', 19: 'Adho Mukha Svanasana', 20: 'Garudasana', 21: 'Sukhasana', 22: 'Astavakrasana', 23: 'Utthita Hasta Padangustasana', 24: 'Uttana Shishosana', 25: 'Utthita Parsvakonasana', 26: 'Utthita Trikonasana', 27: 'Pincha Mayurasana', 28: 'Agnistambhasana', 29: 'Tittibhasana', 30: 'Matsyasana', 31: 'Chaturanga Dandasana', 32: 'Malasana', 33: 'Parighasana', 34: 'Ardha Bhekasana', 35: 'Ardha Matsyendrasana', 36: 'Supta Matsyendrasana', 37: 'Ardha Chandrasana', 38: 'Adho Mukha Vriksasana', 39: 'Ananda Balasana', 40: 'Janu Sirsasana', 41: 'Virasana', 42: 'Krounchasana', 43: 'Utthita Ashwa Sanchalanasana', 44: 'Parsvottanasana', 45: 'Viparita Karani', 46: 'Salabhasana', 47: 'Natarajasana', 48: 'Padmasana', 49: 'Anjaneyasana', 50: 'Marichyasana III', 51: 'Hanumanasana', 52: 'Tadasana', 53: 'Pasasana', 54: 'Eka Pada Rajakapotasana', 55: 'Eka Pada Rajakapotasana II', 56: 'Mayurasana', 57: 'Kapotasana', 58: 'Phalakasana', 59: 'Halasana', 60: 'Eka Pada Koundinyanasana I', 61: 'Eka Pada Koundinyanasana II', 62: 'Marichyasana I', 63: 'Supta Baddha Konasana', 64: 'Supta Padangusthasana', 65: 'Supta Virasana', 66: 'Parivrtta Janu Sirsasana', 67: 'Parivrtta Parsvakonasana', 68: 'Parivrtta Trikonasana', 69: 'Tolasana', 70: 'Paschimottanasana', 72: 'Parsva Bakasana', 73: 'Vasisthasana', 74: 'Anantasana', 75: 'Salamba Bhujangasana', 76: 'Dandasana', 77: 'Uttanasana', 78: 'Ardha Uttanasana', 79: 'Urdhva Prasarita Eka Padasana', 80: 'Salamba Sirsasana', 81: 'Salamba Sarvangasana', 82: 'Vriksasana', 83: 'Urdhva Dhanurasana', 84: 'Dwi Pada Viparita Dandasana', 85: 'Purvottanasana', 86: 'Urdhva Hastasana', 87: 'Urdhva Mukha Svanasana', 88: 'Virabhadrasana I', 89: 'Virabhadrasana II', 90: 'Virabhadrasana III', 91: 'Upavistha Konasana', 92: 'Prasarita Padottanasana', 93: 'Camatkarasana', 94: 'Yoganidrasana', 95: 'Vrischikasana', 96: 'Vajrasana', 97: 'Tulasana', 98: 'Simhasana', 99: 'Makarasana', 100: 'Lolasana', 101: 'Kurmasana', 102: 'Garbha Pindasana', 103: 'Durvasasana', 71: 'Bhujapidasana', 104: 'Bhekasana', 105: 'Bhairavasana', 106: 'Ganda Bherundasana'}
    val_tfms = transforms.Compose([
        Resize(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    test_im_tensor = val_tfms(im)[None]
    res = model_utils.get_yoga_pose_labeled_image(test_im_tensor, model, pose_id_to_name)
    
    return plot_utils.get_yoga_pose_labeled_image(res, im)

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