# -*- coding: utf-8 -*-
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

import cv2
from pose_estimation_lib import model
from pose_estimation_lib import util
from pose_estimation_lib.body import Body
import matplotlib.pyplot as plt
import copy

body_estimation_model = None

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

def load_pose_est_model():
    global body_estimation_model
    if(body_estimation_model == None):
        body_estimation_model = Body('pose_estimation_lib/model_weights/body_pose_model.pth')

def predict_pose(res_url):
    load_pose_est_model()
    global body_estimation_model
    oriImg = cv2.imread('app/'+res_url)
    candidate, subset = body_estimation_model(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    cv2.imwrite('app/'+res_url, canvas)
    return res_url

def get_yoga_pose_labeled_image(im, model):
    pose_id_to_name = {0: 'Bharadvajasana I', 1: 'Padangusthasana', 2: 'Paripurna Navasana', 3: 'Baddha Konasana', 4: 'Dhanurasana', 5: 'Setu Bandha Sarvangasana', 6: 'Ustrasana', 7: 'Marjaryasana', 8: 'Chakravakasana', 9: 'Ashtanga Namaskara', 10: 'Utkatasana', 11: 'Balasana', 12: 'Bhujangasana', 13: 'Savasana', 14: 'Gomukhasana', 15: 'Bitilasana', 16: 'Bakasana', 17: 'Makara Adho Mukha Svanasana', 18: 'Ardha Pincha Mayurasana', 19: 'Adho Mukha Svanasana', 20: 'Garudasana', 21: 'Sukhasana', 22: 'Astavakrasana', 23: 'Utthita Hasta Padangustasana', 24: 'Uttana Shishosana', 25: 'Utthita Parsvakonasana', 26: 'Utthita Trikonasana', 27: 'Pincha Mayurasana', 28: 'Agnistambhasana', 29: 'Tittibhasana', 30: 'Matsyasana', 31: 'Chaturanga Dandasana', 32: 'Malasana', 33: 'Parighasana', 34: 'Ardha Bhekasana', 35: 'Ardha Matsyendrasana', 36: 'Supta Matsyendrasana', 37: 'Ardha Chandrasana', 38: 'Adho Mukha Vriksasana', 39: 'Ananda Balasana', 40: 'Janu Sirsasana', 41: 'Virasana', 42: 'Krounchasana', 43: 'Utthita Ashwa Sanchalanasana', 44: 'Parsvottanasana', 45: 'Viparita Karani', 46: 'Salabhasana', 47: 'Natarajasana', 48: 'Padmasana', 49: 'Anjaneyasana', 50: 'Marichyasana III', 51: 'Hanumanasana', 52: 'Tadasana', 53: 'Pasasana', 54: 'Eka Pada Rajakapotasana', 55: 'Eka Pada Rajakapotasana II', 56: 'Mayurasana', 57: 'Kapotasana', 58: 'Phalakasana', 59: 'Halasana', 60: 'Eka Pada Koundinyanasana I', 61: 'Eka Pada Koundinyanasana II', 62: 'Marichyasana I', 63: 'Supta Baddha Konasana', 64: 'Supta Padangusthasana', 65: 'Supta Virasana', 66: 'Parivrtta Janu Sirsasana', 67: 'Parivrtta Parsvakonasana', 68: 'Parivrtta Trikonasana', 69: 'Tolasana', 70: 'Paschimottanasana', 72: 'Parsva Bakasana', 73: 'Vasisthasana', 74: 'Anantasana', 75: 'Salamba Bhujangasana', 76: 'Dandasana', 77: 'Uttanasana', 78: 'Ardha Uttanasana', 79: 'Urdhva Prasarita Eka Padasana', 80: 'Salamba Sirsasana', 81: 'Salamba Sarvangasana', 82: 'Vriksasana', 83: 'Urdhva Dhanurasana', 84: 'Dwi Pada Viparita Dandasana', 85: 'Purvottanasana', 86: 'Urdhva Hastasana', 87: 'Urdhva Mukha Svanasana', 88: 'Virabhadrasana I', 89: 'Virabhadrasana II', 90: 'Virabhadrasana III', 91: 'Upavistha Konasana', 92: 'Prasarita Padottanasana', 93: 'Camatkarasana', 94: 'Yoganidrasana', 95: 'Vrischikasana', 96: 'Vajrasana', 97: 'Tulasana', 98: 'Simhasana', 99: 'Makarasana', 100: 'Lolasana', 101: 'Kurmasana', 102: 'Garbha Pindasana', 103: 'Durvasasana', 71: 'Bhujapidasana', 104: 'Bhekasana', 105: 'Bhairavasana', 106: 'Ganda Bherundasana'}
    pose_id_to_english_name = {"0":"Bharadvaja’s Twist","1":"Big Toe Pose","2":"Boat Pose","3":"Bound Angle Pose","4":"Bow Pose","5":"Bridge Pose","6":"Camel Pose","7":"Cat Pose","8":"Cat Cow Pose","9":"Knees, Chest and Chin Pose","10":"Chair Pose","11":"Child’s Pose","12":"Cobra Pose","13":"Corpse Pose","14":"Cow Face Pose","15":"Cow Pose","16":"Crane (Crow) Pose","17":"Dolphin Plank Pose","18":"Dolphin Pose","19":"Downward-Facing Dog","20":"Eagle Pose","21":"Easy Pose","22":"Eight-Angle Pose","23":"Extended Hand-To-Big-Toe Pose","24":"Extended Puppy Pose","25":"Extended Side Angle Pose","26":"Extended Triangle Pose","27":"Feathered Peacock Pose","28":"Fire Log Pose","29":"Firefly Pose","30":"Fish Pose","31":"Four-Limbed Staff Pose","32":"Garland Pose","33":"Gate Pose","34":"Half Frog Pose","35":"Half Lord of the Fishes Pose","36":"Lying Half Lord of the Fishes Pose","37":"Half Moon Pose","38":"Handstand","39":"Happy Baby Pose","40":"Head-to-Knee Forward Bend","41":"Hero Pose","42":"Heron Pose","43":"High Lunge","44":"Intense Side Stretch Pose","45":"Legs-Up-the-Wall Pose","46":"Locust Pose","47":"Lord of the Dance Pose","48":"Lotus Pose","49":"Low Lunge","50":"Marichi’s Pose","51":"Monkey Pose","52":"Mountain Pose","53":"Noose Pose","54":"One-Legged King Pigeon Pose","55":"One-Legged King Pigeon Pose II","56":"Peacock Pose","57":"Pigeon Pose","58":"Plank Pose","59":"Plow Pose","60":"Pose Dedicated to the Sage Koundinya I","61":"Pose Dedicated to the Sage Koundinya II","62":"Pose Dedicated to the Sage Marichi I","63":"Reclining Bound Angle Pose","64":"Reclining Hand-to-Big-Toe Pose","65":"Reclining Hero Pose","66":"Revolved Head-to-Knee Pose","67":"Revolved Side Angle Pose","68":"Revolved Triangle Pose","69":"Scale Pose","70":"Seated Forward Bend","71":"Bhujapidasana","72":"Side Crane (Crow) Pose","73":"Side Plank Pose","74":"Side-Reclining Leg Lift","75":"Sphinx Pose","76":"Staff Pose","77":"Standing Forward Bend","78":"Standing Half Forward Bend","79":"Standing Split","80":"Supported Headstand","81":"Supported Shoulderstand","82":"Tree Pose","83":"Upward Bow (Wheel) Pose","84":"Upward Facing Two-Foot Staff Pose","85":"Upward Plank Pose","86":"Upward Salute","87":"Upward-Facing Dog Pose","88":"Warrior I Pose","89":"Warrior II Pose","90":"Warrior III Pose","91":"Wide-Angle Seated Forward Bend","92":"Wide-Legged Forward Bend","93":"Wild Thing","94":"Yogic Sleep","95":"Scorpion Pose","96":"Thunderbolt Pose","97":"Balance Pose","98":"Lion Pose","99":"Crocodile Pose","100":"Pendant Pose","101":"Tortoise Pose","102":"Embryo in Womb Pose","103":"Durvasasana","104":"Frog Pose","105":"Formidable Pose","106":"Formidable Face/Chin Stand Pose"}
    val_tfms = transforms.Compose([
        Resize(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    test_im_tensor = val_tfms(im)[None]
    res = model_utils.get_yoga_pose_labeled_image(test_im_tensor, model, pose_id_to_name, pose_id_to_english_name)

    res_url = plot_utils.get_yoga_pose_labeled_image(res, im) 
    res_url = predict_pose(res_url)
    return res_url

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