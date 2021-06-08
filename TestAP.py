import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as transforms
import math
import torchvision
import csv
import random
import cv2
import time
import sys
import json
import argparse
import os
from math import pi
from PIL import Image
from torch.autograd import Variable
from canny_edge_detector import cannyEdgeDetector
from skimage.measure import approximate_polygon, find_contours
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from wisdmapi import wisdm_getlatlon
from wisdmapi import wisdm_checkavi
from googleapi import google_getaerial
from googleapi import google_avimarker
from bingapi import bing_getaerial
from bingapi import bing_avimarker

# Get inputs
parser = argparse.ArgumentParser()
parser.add_argument("-P", "--postcode", help="postcode in format YO105DD")
parser.add_argument("-A", "--address", help="address in format '2 CRANBROOK AVENUE'")
parser.add_argument("-L", "--latlon", help="latlon in format '53.964962, -1.124317', if "
                                           "latlon and address are both given, latlon overides address")
parser.add_argument("-W", "--wisdmcheck", choices=['Y', 'N'], default='Y',
                    help="run wisdm availability check or not, default is Y")
args = parser.parse_args()
addr = str(args.address)
postcode = str(args.postcode)
center_latlon = args.latlon
if addr is None or postcode is None and center_latlon is None:
    raise NameError("Please provide address+postcode or latlon")

st = time.time()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
loader = transforms.Compose([transforms.ToTensor()])
zoom = 20 # map zoom level
map_width = 600 # map pixel size
map_height = 600


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# compute image gradient
def gradient_img(img):
    img = img.squeeze(0)
    ten = torch.unbind(img)
    x = ten[0].unsqueeze(0).unsqueeze(0)

    a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))
    G_x = conv1(Variable(x)).data.view(1, x.shape[2], x.shape[3])

    b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight = nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))
    G_y = conv2(Variable(x)).data.view(1, x.shape[2], x.shape[3])

    G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
    # theta = np.arctan2(G_y, G_x)
    return G


def get_model_instance_segementation(num_classes):
    # load an instance model pretrained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # get number of input features from classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pretrained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # get number of input features from the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


# convert lat lon to pixel number
def LatLontoXY(lat_center,lon_center,zoom):
    C =(256/(2*pi) )* 2**zoom

    x=C*(math.radians(lon_center)+pi)
    y=C*(pi-math.log( math.tan(  (pi/4) + math.radians(lat_center)/2    )  ))

    return x,y


# convert pixel relative to center to latlon
def xy2LatLon(lat_center,lon_center,zoom,width_internal,height_internal,pxX_internal,pxY_internal):

    xcenter,ycenter=LatLontoXY(lat_center,lon_center,zoom)

    xPoint=xcenter- (width_internal/2-pxX_internal)
    ypoint=ycenter -(height_internal/2-pxY_internal)


    C = (256 / (2 * pi)) * 2 ** zoom
    M = (xPoint/C)-pi
    N =-(ypoint/C) + pi

    lon_Point =math.degrees(M)
    lat_Point =math.degrees( (math.atan( math.e**N)-(pi/4))*2 )

    return lat_Point,lon_Point


def pixelsize(lat_center, zoom):
    size = 156543.03392 * math.cos(math.radians(lat_center)) / math.pow(2, zoom)
    return size


def extract(list, idx):   # to extract the idxth item in all sublists
    return [item[idx] for item in list]


# get center lat lon if only address+post code is given
if center_latlon is None:
    center_latlon = wisdm_getlatlon(postcode, addr)

index = center_latlon.index(',')
lat_center = np.float(center_latlon[0:index])
lon_center = np.float(center_latlon[index+2:-1])

# get map image

if not os.path.exists('./mapsamples'):
    os.makedirs('./mapsamples')
bing_getaerial(center_latlon)
img = Image.open('./mapsamples/testmap.png')
img = loader(img.convert('RGB'))
num_classes = 2

# load model from a trained network
model = get_model_instance_segementation(num_classes)

model.load_state_dict(torch.load('MRCNNRes50(COCO).pth', map_location=device))
# let the model know the number of detections per image
model.roi_heads.detections_per_img = 3
model.eval()
model.to(device)
with torch.no_grad():
    prediction = model([img.to(device)])

img1 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())


num_obj = len(prediction[0]['masks'][:, 0])
# fig, axs = plt.subplots(2, num_obj+1, figsize=(20, 10))
# axs[0, 0].imshow(img1)
# axs[0, 0].set_title('original image')

mask_identified = 0

for idx in range(1, num_obj+1):
    img3 = Image.fromarray(prediction[0]['masks'][idx-1, 0].mul(255).byte().cpu().numpy())
    img_t = np.array(img3)
    ret, img3 = cv2.threshold(img_t, 200, 255, cv2.THRESH_TOZERO)
    if img_t[300, 300] != 0 and mask_identified == 0:
        imgx = img3
        mask_identified = 1
    # axs[0, idx].imshow(img3)
    # axs[0, idx].set_title('mask%d' % idx)
    img3 = loader(img3).unsqueeze(0)
    img3.to(device, torch.float)
    grad = gradient_img(img3)
    grad = grad.detach().cpu().squeeze(0)

    # axs[1, idx].imshow(grad)
    # axs[1, idx].set_title('gradient%d' % idx)

to_grayscale = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                   transforms.ToTensor()])

img1_grayscale = to_grayscale(img1)
imgx_rgb = Image.fromarray(imgx).convert('RGB')
imgx = loader(imgx_rgb)
edge = cannyEdgeDetector(imgx).detect()
edge = edge[0]

# find contours and generate polygon using skimage
contours = find_contours(edge, 0, 'high')
result_polygon = np.zeros(edge.shape + (3, ), np.uint8)
for contour in contours:
    polygon = approximate_polygon(contour, tolerance=1)
    polygon = polygon.astype(np.int).tolist()
    for idx, coords in enumerate(polygon[:-1]):
        y1, x1, y2, x2 = coords + polygon[idx + 1]
        result_polygon = cv2.line(result_polygon, (x1, y1), (x2, y2), (0, 255, 0), 1)

img1_np = np.array(img1)
result_polygon = Image.fromarray(result_polygon)
result_polygon.save('mapsamples/edge.png','png')
imgx_rgb.save('mapsamples/mask.png','png')
result_polygon_gray = to_grayscale(result_polygon)
result_polygon_np = np.array(result_polygon_gray)
edge_idx = np.where(result_polygon_np.squeeze() != 0)
edge_t = list(zip(*edge_idx))  # 'transpose' the tuple
edge_idx_random = random.sample(edge_t, 50)   # randomly pick some edge pixels
edge_latlons = []
# find latlons of the edge pixels
for edge_y, edge_x in edge_idx_random:
    lat, lon = xy2LatLon(lat_center, lon_center, zoom, map_width, map_height, edge_x, edge_y)
    latlon = str(lat)+','+str(lon)
    edge_latlons.append(latlon)
# get pixel size
pixel_size = pixelsize(lat_center, zoom)
dist_pixel = 1 # in meters, defines the size of the box to choose the next pixel
dist_num_pixel = math.ceil(dist_pixel / pixel_size) # convert to number of pixels
# randomly pick 1 pixel to start
edge_idx_random = random.sample(edge_t, 1)


avi_all = []
avi_comb_all = []
ap_names_all = []
inference_time = time.time() - st
print('Time before calling wisdm availability check is ', inference_time)
if args.wisdmcheck == 'N':
    json_string = json.dumps([{'Edge_latlon': edge_latlon}
                              for edge_latlon in edge_latlons])
    with open('./mapsamples/Latlon_nocheck.txt', 'w') as file:  # save the dict to file
        file.write(json_string)

    dn = os.path.dirname(os.path.realpath(__file__))
    dirname = str(lat_center) + '_' + str(lon_center)
    if not os.path.exists('./' + dirname):
        os.makedirs('./' + dirname)
    os.system('mv ' + './mapsamples/Latlon_nocheck.txt ' + dirname)

    sys.exit('Availability is not checked by WISDM, dump edge latlons')

for latlon in edge_latlons:
    ap_names, avis, avi_comb = wisdm_checkavi(latlon)
    avi_all.append(avis)
    ap_names_all.append(ap_names)
    avi_comb_all.append(avi_comb)
    time.sleep(0.5)   # wisdm crashes if the api is request too frequently

bing_avimarker(center_latlon, edge_latlons, avi_all, avi_comb_all, ap_names_all)
num_aps = len(avi_all[0])
for idx in range(num_aps):
    json_string = json.dumps([{'Edge_latlon': edge_latlon, 'Result': avi}
                              for edge_latlon, avi in zip(edge_latlons, extract(avi_all, idx))])
    # save result image for each AP
    with open('./mapsamples/Latlon_result_' + ap_names_all[0][idx] + '.txt', 'w') as file:  # save the dict to file
        file.write(json_string)
json_string = json.dumps([{'Edge_latlon': edge_latlon, 'Result': avi}
                         for edge_latlon, avi in zip(edge_latlons, avi_comb_all)])
with open('./mapsamples/Latlon_result_allAPs.txt', 'w') as file:  # save the dict to file
    file.write(json_string)

dirname = str(lat_center) + '_' + str(lon_center)
if not os.path.exists('./' + dirname):
    os.makedirs('./' + dirname)
os.system('mv ' + './mapsamples/testmap* ' + dirname)
os.system('mv ' + './mapsamples/Latlon* ' + dirname)

