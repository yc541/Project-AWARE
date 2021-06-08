import os
import torch
import numpy as np
import torchvision
import transforms as T
import utils
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
from torch.utils.data.dataloader import DataLoader


class GooglemapDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all images, sorting to ensure they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "mapimages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "mapmasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "mapimages", self.imgs[idx])
        mask_path = os.path.join(self.root, "mapmasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # mask is not converted to RGB
        mask = Image.open(mask_path)
        # convert PIL image into numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background and remove it
        obj_ids = obj_ids[1:]
        # split the color coded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only 1 class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


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


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


if __name__ == '__main__':

    # train on GPU if possible
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    dataset = GooglemapDataset('mapsamples', get_transform(train=True))
    dataset_test = GooglemapDataset('mapsamples', get_transform(train=False))
    # split data set into train and test
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])  # everything except the last 50
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])  # the last 50
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
    # get the model using our helper function
    model = get_model_instance_segementation(num_classes)
    # modify hyper parameters if required
    # batch_size_per_image = 2048
    # model.roi_heads.batch_size_per_image = batch_size_per_image
    # positive_fraction = 0.25
    # model.roi_heads.positive_fraction = positive_fraction
    # move model to device
    model.to(device)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(params, lr=0.005, betas=(0.5, 0.999), weight_decay=0.0005)
    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # training
    num_epochs = 10
    for epoch in range(num_epochs):
        # train for one epoch, print every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update lr
        lr_scheduler.step()
        # evaluate
        history = evaluate(model, data_loader_test, device=device)
        # print(history)
    torch.save(model.state_dict(), 'MRCNN.pth')
    torch.save(history, 'History.pth')
