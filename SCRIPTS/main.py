from SCRIPTS import data_preparation
from SCRIPTS import data_representation
from SCRIPTS import data_query
from SCRIPTS import OPPD_utils

import cv2

if __name__ == '__main__':



    data_path = "/Users/laurenzohnemuller/DATA_PlantIdentification/images_full"
    all_images = data_preparation.all_images_of_path(data_path)

    # combine all annotations in a single file
    combined_anno = data_preparation.combine_anno(all_images)

    #data_query.find_image_by_ID(281427, combined_anno)
    #data_query.find_image_by_EPPO("SINAR ", combined_anno)

    # get all categories -> all plant types
    categories = data_preparation.get_categories(combined_anno)
    print(categories)

    # print the distribution of the categories in the data
    #data_representation.class_distribution(combined_anno)
    data_query.find_small_BB(combined_anno)
    '''
    # split data into training, validation and testing
    train_plants, val_plants, test_plants = data_preparation.split_data(combined_anno)

    data_representation.plant_distribution(train_plants, "train")
    data_representation.plant_distribution(val_plants, "val")
    data_representation.plant_distribution(test_plants, "test")
    
    # transform each set to darknet format
    data_preparation.transform_to_darknet(train_plants, categories, 'train')
    data_preparation.transform_to_darknet(val_plants, categories, 'val')
    data_preparation.transform_to_darknet(test_plants, categories, 'test')
    # Note to see yolov4-csp architecture, write in terminal: "cat /Users/laurenzohnemuller/PycharmProjects/PlantIdentification/ScaledYOLOv4/models/yolov4-csp.yaml"
    '''