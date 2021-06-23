from SCRIPTS import data_preparation
from SCRIPTS import data_representation
from SCRIPTS import data_query
from SCRIPTS import OPPD_utils

import cv2
import os

if __name__ == '__main__':
    path = '/Users/laurenzohnemuller/PycharmProjects/PlantIdentification/plants/images/train'

    '''
    lstLabels = []
    path =  "/Users/laurenzohnemuller/PycharmProjects/PlantIdentification/plants/labels/train"
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith("txt"):
                lstLabels.append((root + '/'+ name))
    print("Amount of labels:", len(lstLabels))
    count = 0
    for label in lstLabels:
        with open(label) as f:
            lines = f.readlines()

            if len(lines) == 0:
                count += 1

    print(count)
    '''

    data_path = "/Users/laurenzohnemuller/DATA_PlantIdentification/images_full"

    all_images = data_preparation.all_images_of_path(data_path)

    # combine all annotations in a single file
    combined_anno = data_preparation.combine_anno(all_images)

    # remove annotations with too small bounding boxes
    data_preparation.remove_small_bbox(combined_anno)

    # data_query.get_stats_about_eppo(combined_anno, "GERMO ")
    # data_query.find_image_by_ID(283922, combined_anno)
    # data_query.find_image_by_EPPO("EPHHE ", combined_anno)

    # get all categories -> all plant types
    categories = data_preparation.get_categories(combined_anno)
    print(categories)

    # print the distribution of the categories in the data
    amt_objects = data_representation.class_distribution(combined_anno)

    # split data into training, validation and testing
    train_plants, val_plants, test_plants = data_preparation.split_data(combined_anno)

    data_representation.plant_distribution(train_plants, "train")
    data_representation.plant_distribution(val_plants, "val")
    data_representation.plant_distribution(test_plants, "test")


    # img_train_path = "/Users/laurenzohnemuller/PycharmProjects/PlantIdentification/plants/images/train"
    # img_test_path = "/Users/laurenzohnemuller/PycharmProjects/PlantIdentification/plants/images/test"

    # very inefficient for a lot of images, thats why commented
    # data_preparation.remove_duplicates(img_train_path, img_test_path)

    # uncomment for no pre-processed data
    '''
    data_preparation.no_preprocessing(train_plants, categories, 'train')
    data_preparation.no_preprocessing(val_plants, categories, 'val')
    data_preparation.no_preprocessing(test_plants, categories, 'test')
    
  '''
    # transform each set to darknet format
    data_preparation.transform_to_darknet(train_plants, categories, 'train')
    data_preparation.transform_to_darknet(val_plants, categories, 'val')
    data_preparation.transform_to_darknet(test_plants, categories, 'test')


    # test accuracy after model training
    accuracy = [0.68, 0.856, 0.519, 0.783, 0.654, 0.688, 0.82, 0.763, 0.964, 0.681, 0.716, 0.915, 0.978, 0.755, 0.884,
                0.958, 0.395, 0.918, 0.976, 0.896, 0.657, 0.796, 0.431, 0.255]

    data_representation.scatter_plot(amt_objects, accuracy, categories)