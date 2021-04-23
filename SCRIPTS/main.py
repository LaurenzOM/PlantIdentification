from SCRIPTS import data_preparation
from SCRIPTS import OPPD_utils
import time

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
