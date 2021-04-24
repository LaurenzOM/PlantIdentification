from SCRIPTS import data_preparation
from SCRIPTS import OPPD_utils
from ScaledYOLOv4 import models
import time
import torch
import PIL.Image as Image


if __name__ == '__main__':

    data_path = "/Users/laurenzohnemuller/PycharmProjects/PlantIdentification/DATA"
    start = time.time()
    all_images = data_preparation.all_images_of_path(data_path)
    end = time.time()
    print(end - start)
    #data_preparation.addBndBox2All(all_images)
    combined_anno = data_preparation.combine_anno(all_images)
    categories = data_preparation.getCategories(combined_anno)
    train_plants, val_plants = data_preparation.split_data(combined_anno)
    data_preparation.transform_to_darknet(train_plants, categories, 'train')
    data_preparation.transform_to_darknet(val_plants, categories, 'val')

# Note to see yolov4-csp architecture, write in terminal: "cat /Users/laurenzohnemuller/PycharmProjects/PlantIdentification/ScaledYOLOv4/models/yolov4-csp.yaml"
