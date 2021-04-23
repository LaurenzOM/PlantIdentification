from SCRIPTS import OPPD_utils
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np
import urllib
import PIL.Image as Image


def all_images_of_path(data_path: str) -> [str, str]:
    return OPPD_utils.getImagesInFolder(data_path)


def addBndBox2All(all_images: [str, str]) -> [dict, dict]:
    """
    This method is used to add bounding boxes to all images. It saves the images in a folder, as they are later
    needed for training.
    :param: all_images, which is a tuple [folder, filename]
    :return: a dictionary with an entry for each image, with the image id as the key and the image as an ndarray with
    bounding boxes as the element

    TODO: can later be stored in a csv file
    """
    all_images_bndbox = {}
    all_images_dict = {}
    count = 0
    # iterate over all images
    for img in all_images:
        count = count + 1
        print(count)
        # generate the path to each image
        path_to_image = img[0] + "/" + img[1]
        # get image id and image with bounding boxes
        img_id, img_bndbox = OPPD_utils.addBndBoxes2Image(path_to_image)
        all_images_bndbox[img_id] = img_bndbox

        # dictionary of images without bounding boxes (also key:img_id; element: image)
        all_images_dict[img_id] = img

        # create a unique title for each image: "[image_id].jpg"
        title = str(img_id) + ".jpg"
        # path where the image is stored
        # TODO: at the moment the path leads always to ALOMY, needs to be plant specfic
        path = "/Users/laurenzohnemuller/PycharmProjects/PlantIdentification/images_full_bndbox"
        # generate image and store in path
        # Note: we don't need to store the images as actual images, the ndarrays will be enough (the matrix will be the
        # input for later models anyways)
        # cv2.imwrite(os.path.join(path, title), img_bndbox)

    return all_images_dict, all_images_bndbox


def combine_anno(all_images: list) -> list:
    """
    Method to combine all the annotations.
    :param all_images: A tuple of all images -> [fodler, filename]
    :return: a list, which includes all annotations
    """
    all_anno = []

    for img in all_images:
        path_to_img = img[0] + "/" + img[1]
        path_anno = path_to_img.replace('.jpg', '.json')
        all_anno.append(OPPD_utils.readJSONAnnotation(path_anno))

    # different plants in images
    plant = []
    for a in all_anno:
        for p in a['plants']:
            plant.append(p['eppo'])
    categories = list(set(plant))
    categories.sort()
    print(categories)

    return all_anno


def getCategories(all_anno: list) -> list:
    # different plants in images
    plant = []
    for a in all_anno:
        for p in a['plants']:
            plant.append(p['eppo'])
    categories = list(set(plant))
    categories.sort()
    print(categories)
    return categories


def split_data(all_anno: list) -> [list, list]:
    """
    Method to split the data into training and validation set. Split 90/10
    :param all_anno:
    :return: a tuple of list [training set, validation set]
    """
    train_plants, val_plants = train_test_split(all_anno, test_size=0.1)
    print(len(train_plants), len(val_plants))
    return [train_plants, val_plants]

def transform_to_darknet(all_anno: list, categories: list, dataset_type: str):

    images_path = Path(f"plants/images/{dataset_type}")
    images_path.mkdir(parents=True, exist_ok=True)

    labels_path = Path(f"plants/labels/{dataset_type}")
    labels_path.mkdir(parents=True, exist_ok=True)

    path = "/Users/laurenzohnemuller/PycharmProjects/PlantIdentification/DATA/images_full/ALOMY"

    # iterate through all plant annotations
    for img_id, row in enumerate(tqdm(all_anno)):
        img_id = row["image_id"]
        image_name = f"{img_id}.jpeg"
        img = path+"/"+row["filename"]
        img = Image.open(img)
        img = img.convert("RGB")
        width, height = img.size
        img.save(str(images_path / image_name), "JPEG")
        label_name = f"{img_id}.txt"
        with (labels_path / label_name).open(mode="w") as label_file:

            for plant in row['plants']:

                label = plant["eppo"]
                category_idx = categories.index(label)
                bndbox = plant['bndbox']
                x_min, x_max = bndbox["xmin"], bndbox["xmax"]
                y_min, y_max = bndbox["ymin"], bndbox["ymax"]

                # normalizing coordinates
                # TODO: not sure if done correctly
                x_min, x_max = x_min / width, x_max / width
                y_min, y_max = y_min / height, y_max / height

                bbox_width = x_max - x_min
                bbox_height = y_max - y_min

                label_file.write(
                    f"{category_idx} {x_min + bbox_width / 2} {y_min + bbox_height / 2} {bbox_width} {bbox_height}\n"
                )
