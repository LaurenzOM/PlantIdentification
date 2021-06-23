import os

from SCRIPTS import OPPD_utils
from SCRIPTS import data_query
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

from PIL import Image, ImageEnhance, ImageStat

import cv2
import numpy as np
import urllib

"""
This file is used to prepare the original data in such a way that it can be used by the Scaled-YOLOv4 model. This also 
contains data preprocessing. 
"""


def all_images_of_path(data_path: str) -> [str, str]:
    return OPPD_utils.getImagesInFolder(data_path)


def addBndBox2All(all_images: [str, str]) -> [dict, dict]:
    """
    This method is used to add bounding boxes to all images. It saves the images in a folder, as they are later
    needed for training.
    :param: all_images, which is a tuple [folder, filename]
    :return: a dictionary with an entry for each image, with the image id as the key and the image as an ndarray with
    bounding boxes as the element

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
        cv2.imwrite(os.path.join(path, title), img_bndbox)

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
    return all_anno


def get_categories(all_anno: list) -> list:
    """
    This method is used to get all types of plants in the data. Those types of plants are referred as categories.
    :param all_anno: a list of all annotations
    :return: a list of categories
    """
    # different plants in images
    plant = []
    for a in all_anno:
        for p in a['plants']:
            plant.append(p['eppo'])
            if p["eppo"] == '3UNCLK':
                print(a, p)
    categories = list(set(plant))
    categories.sort()
    return categories


def split_data(all_anno: list) -> [list, list, list]:
    """
    Method to split the data into training and validation set. Split 90/10
    :param all_anno:  list of all annotations
    :return: a list of lists [training set, validation set, testing set]
    """
    train_plants, test_plants = train_test_split(all_anno, test_size=0.15, random_state=30)
    train_plants, val_plants = train_test_split(train_plants, test_size=0.15, random_state=30)
    print("size of testing set: ", len(test_plants), " | size of training set: ", len(train_plants),
          " | size of validation set: ", len(val_plants))
    return [train_plants, val_plants, test_plants]


def transform_to_darknet(all_anno: list, categories: list, dataset_type: str):
    """
    This method transforms the annotated image data into the darknet transforms. Note that the bounding box coordinates
    are normalized. Before transforming to the darknet format, the black background of the images are removed by
    thresholding and cropping the image. Here the bounding box coordinates need to be adjusted as well. Additionally,
    the images and the box coordinates are resized to a samller size for efficiency.

    :param all_anno: a list of annotations of [multiple] image(s)
    :param categories: a list of all categories/labels
    :param dataset_type: a string that describes the set: either training, validation or testing
    :return: stores pre-processed images and annotations
    """
    images_path = Path(f"plants/images/{dataset_type}")
    images_path.mkdir(parents=True, exist_ok=True)

    labels_path = Path(f"plants/labels/{dataset_type}")
    labels_path.mkdir(parents=True, exist_ok=True)

    path = "/Users/laurenzohnemuller/DATA_PlantIdentification/images_full"

    # factor by which the images are resized
    scale_factor = 5

    # iterate through all plant annotations
    for img_id, row in enumerate(tqdm(all_anno)):
        # create image path and open image
        img_id = row["image_id"]
        image_name = f"{img_id}.jpeg"
        img_path = path + "/" + row["filename"]
        img = cv2.imread(img_path)

        # remove black parts of image:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # apply Otsu threshold
        threshold, img_threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        # create bounding box around plant
        bbox = cv2.boundingRect(img_threshold)
        x, y, width, height = bbox
        # crop image, such that black background is removed
        img_foreground = img[y:y + height, x:x + width]

        # resize image by scale_factor | and transformed cv2 image to PIL and bgr to rgb
        img_foreground = Image.fromarray(img_foreground)
        R, G, B = img_foreground.split()
        img_foreground = Image.merge("RGB", (B, G, R))
        width, height = img_foreground.size
        width, height = int(width / scale_factor), int(height / scale_factor)
        new_size = (width, height)
        img_foreground = img_foreground.resize(new_size, 0)

        # apply a high contrast to the image
        img_foreground_contrast = contrast(img_foreground)

        img_foreground_contrast.save(str(images_path / image_name), "JPEG")

        label_name = f"{img_id}.txt"
        with (labels_path / label_name).open(mode="w") as label_file:

            # iterate over each plant in image
            for plant in row['plants']:
                label = plant["eppo"]
                category_idx = categories.index(label)
                bndbox = plant['bndbox']
                # adjust coordinates after cropping image and after resizing
                x_min, x_max = int((bndbox["xmin"] - x) / scale_factor), int((bndbox["xmax"] - x) / scale_factor)
                y_min, y_max = int((bndbox["ymin"] - y) / scale_factor), int((bndbox["ymax"] - y) / scale_factor)

                # normalizing coordinates
                x_min, x_max = x_min / width, x_max / width
                y_min, y_max = y_min / height, y_max / height

                bbox_width = x_max - x_min
                bbox_height = y_max - y_min

                label_file.write(
                    f"{category_idx} {x_min + bbox_width / 2} {y_min + bbox_height / 2} {bbox_width} {bbox_height}\n"
                )

def no_preprocessing(all_anno: list, categories: list, dataset_type: str):

    """
    This method resembles the transform_to_darknet method. However, in this case no cropping of the images is done.
    They are still resized to increase download/upload speed and contrast is still added.
    :param all_anno: a list of annotations of [multiple] image(s)
    :param categories: a list of all categories/labels
    :param dataset_type: a string that describes the set: either training, validation or testing
    :return: stores pre-processed images and annotations
    """

    images_path = Path(f"plants_no_pp/images/{dataset_type}")
    images_path.mkdir(parents=True, exist_ok=True)

    labels_path = Path(f"plants_no_pp/labels/{dataset_type}")
    labels_path.mkdir(parents=True, exist_ok=True)

    path = "/Users/laurenzohnemuller/DATA_PlantIdentification/images_full"

    # factor by which the images are resized
    scale_factor = 7

    # iterate through all plant annotations
    for img_id, row in enumerate(tqdm(all_anno)):
        # create image path and open image
        img_id = row["image_id"]
        image_name = f"{img_id}.jpeg"
        img_path = path + "/" + row["filename"]
        img = Image.open(img_path)

        # resize image by scale_factor
        width, height = img.size
        width, height = int(width / scale_factor), int(height / scale_factor)
        new_size = (width, height)
        img = img.resize(new_size, 0)

        # apply a high contrast to the image
        img_contrast = contrast(img)
        img_contrast.save(str(images_path / image_name), "JPEG")

        label_name = f"{img_id}.txt"
        with (labels_path / label_name).open(mode="w") as label_file:

            # iterate over each plant in image
            for plant in row['plants']:
                label = plant["eppo"]
                category_idx = categories.index(label)
                bndbox = plant['bndbox']
                # adjust coordinates after cropping image and after resizing
                x_min, x_max = int((bndbox["xmin"]) / scale_factor), int((bndbox["xmax"]) / scale_factor)
                y_min, y_max = int((bndbox["ymin"]) / scale_factor), int((bndbox["ymax"]) / scale_factor)

                # normalizing coordinates
                x_min, x_max = x_min / width, x_max / width
                y_min, y_max = y_min / height, y_max / height

                bbox_width = x_max - x_min
                bbox_height = y_max - y_min

                label_file.write(
                    f"{category_idx} {x_min + bbox_width / 2} {y_min + bbox_height / 2} {bbox_width} {bbox_height}\n"
                )

def contrast(img):
    """
    This method adds contrast to a given image with a factor of 1.3, where 1 is adding no contrast.
    :param img: The given image, which is a PIL Image Object
    :return: a PIL Image object with added contrast
    """

    # image brightness enhancer
    enhancer = ImageEnhance.Contrast(img)

    # contrast factor
    factor = 1.4
    img_contrast = enhancer.enhance(factor)
    return img_contrast


def remove_small_bbox(all_anno: list):
    """
    This method was used to find extremely small bounding boxes. Those were probably caused by an annotation error. Note
    that this method checks for small bounding boxes before resizing
    :param all_anno:
    :return:
    """

    path = "/Users/laurenzohnemuller/DATA_PlantIdentification/images_full/"
    print("The following plants were removed in the corresponding file since there bbox were too small:")
    filenames = {}
    # iterate over all iterations
    for anno in all_anno:

        for plant in anno["plants"]:
            bbox = plant["bndbox"]
            width = bbox["xmax"] - bbox["xmin"]
            height = bbox["ymax"] - bbox["ymin"]
            # width less than 9 pixels
            if width < 15 and height < 15:
                print("filename:", anno["filename"], "| plant_id", plant["plant_id"])
                filenames[anno["filename"]] = plant["bndbox_id"]
                anno["plants"].remove(plant)

    '''    
    for file in filenames:
        id, output = data_query.show_bndbox_of_ID(path + file, filenames[file])
        cv2.imshow("file", output)
        cv2.waitKey(0)
    '''
    return all_anno


def remove_duplicates(img_path1: str, img_path2: str):
    """
    Method to search for duplicate images in two different paths by using the pixel mean.
    :param img_path1: Image path one
    :param img_path2: Image path two that the images are compared two
    :return:
    """
    img_files1 = [_ for _ in os.listdir(img_path1) if _.endswith("jpeg")]
    img_files2 = [_ for _ in os.listdir(img_path2) if _.endswith("jpeg")]

    duplicates = []

    for img_file in tqdm(img_files1):
        if not img_file in duplicates:
            # convert to grayscale for efficiency
            img = Image.open(os.path.join(img_path1, img_file)).convert('LA')
            pixel_mean = ImageStat.Stat(img).mean

            for img_file_check in img_files2:
                if img_file != img_file_check:
                    img_check = Image.open(os.path.join(img_path2, img_file_check)).convert('LA')
                    pixel_mean_check = ImageStat.Stat(img_check).mean

                    if pixel_mean == pixel_mean_check:
                        duplicates.append(img_file)
                        duplicates.append(img_file_check)

    print(duplicates)
