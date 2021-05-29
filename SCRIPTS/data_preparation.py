import os

from SCRIPTS import OPPD_utils
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

from PIL import Image, ImageEnhance

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
    train_plants, test_plants = train_test_split(all_anno, test_size=0.1)
    train_plants, val_plants = train_test_split(train_plants, test_size=0.2)
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

def contrast(img):
    """
    This method adds contrast to a given image with a factor of 1.5, where 1 is adding no contrast.
    :param img: The given image, which is a PIL Image Object
    :return: a PIL Image object with added contrast
    """

    # image brightness enhancer
    enhancer = ImageEnhance.Contrast(img)

    # contrast factor
    factor = 1.5
    img_contrast = enhancer.enhance(factor)
    return img_contrast

'''
def contrast():
    data_path = "/Users/laurenzohnemuller/PycharmProjects/PlantIdentification/plants/images/test/285444.jpeg"

    matplotlib.rcParams['font.size'] = 9

    # Load an example image
    img = io.imread(data_path)
    img = rgb2gray(img)
    img = img_as_ubyte(img)

    # Global equalize
    img_rescale = exposure.equalize_hist(img)

    # Equalization
    selem = disk(30)
    img_eq = rank.equalize(img, selem=selem)

    # Display results
    fig = plt.figure(figsize=(8, 5))
    axes = np.zeros((2, 3), dtype=np.object)
    axes[0, 0] = plt.subplot(2, 3, 1)
    axes[0, 1] = plt.subplot(2, 3, 2, sharex=axes[0, 0], sharey=axes[0, 0])
    axes[0, 2] = plt.subplot(2, 3, 3, sharex=axes[0, 0], sharey=axes[0, 0])
    axes[1, 0] = plt.subplot(2, 3, 4)
    axes[1, 1] = plt.subplot(2, 3, 5)
    axes[1, 2] = plt.subplot(2, 3, 6)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    ax_img.set_title('Low contrast image')
    ax_hist.set_ylabel('Number of pixels')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
    ax_img.set_title('Global equalise')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
    ax_img.set_title('Local equalize')
    ax_cdf.set_ylabel('Fraction of total intensity')

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()

def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins)
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')

    xmin, xmax = dtype_range[image.dtype.type]
    ax_hist.set_xlim(xmin, xmax)

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')

    return ax_img, ax_hist, ax_cdf
'''



