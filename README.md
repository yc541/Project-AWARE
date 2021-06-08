# Project-AWARE
Building edge extraction from aerial images using Mask R-CNN, to assist wireless network coverage planning tools to find potential CPE antenna mounting locations. The project uses WISDM (https://www.wirelesscoverage.com/) wireless coverage planning system to produce wireless coverage reports using the building edge locations extracted by Mask R-CNN. Any other wireless coverage planning tools which are capable of generating point-to-point path profiles should be compatible to this project. 

The Mask R-CNN building edge detection part is built upon the torchvision object detection fine tuning tutorial, available at 
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

The project can run under the conda environment, with a few required packages:

    pytorch, torchvision, opencv, PIL, skimage
    
Some other packages may need to be installed as well depending on the local environment, the import list in TestAP.py should have some hints on that.

We have already trained a Mask R-CNN using an aerial image dataset (200 images) including rural and suburban residential properties in North Yorkshire, UK. The data set is available at: 

https://drive.google.com/file/d/1KKwxUfkRL3GYmyULLFLof0SDHBf11W1C/view?usp=sharing

Here are some samples:
![Fig5](https://user-images.githubusercontent.com/8125847/121216593-f4957b80-c878-11eb-853b-1a3a33cd6545.png)


Extract the dataset to your local folder if you would like to train with our dataset.

To train a Mask R-CNN use:

    python3 train.py
    
Or you can add your own dataset (images and masks) into the folders /mapsamples/mapimages and /mapsamples/mapmasks.

We use Bingmap to acuiqre aerial image of the target property, you will need a valid Bingmap API key to download aerial images. If you have a valid key, modify bingapi.py with your own key.


The main script to extract building edges and produce wireless coverage maps is TestAP.py, help can be brought out by:

    python3 TestAP.py -h
    
-P (postcode) and -A (address) should be given at the same time and the address needs to match one of the outputs of wisdm api of that postcode. Use this option only if you have access to WISDM!!!

if -L (latlon) is given, the latitude and longitude will be used as the map center point directly without calling wisdm api to search the addresses. We recommand using this option if you do not have WISDM. If -L is given, the location from -L will override -P and -A.

if -W is N (default is Y), wisdm network availability checker api will not be called, there will be a folder named with the center point latlon, inside the folder is a list of 50 edge point latlons in json format. We recommand using this option if you do not have WISDM. 

if -W is Y, wisdm anetwork vailability checker api will be called for all 50 edge points, and in the folder there will be the original map and the avaiability images for APs, and also the individual latlon availability check results in json format. Use this option only if you have access to WISDM!!!


Here are some building edge detection results using our trained Mask R-CNN on residential properties:
![Fig7](https://user-images.githubusercontent.com/8125847/121217169-7be2ef00-c879-11eb-92e0-18356b8a4ab9.png)

Here are some examples of the coverage maps generated using the Mask R-CNN building detection and path profile results from WISDM:
![Fig8](https://user-images.githubusercontent.com/8125847/121217618-f01d9280-c879-11eb-8421-70dbd84dbf42.png)


