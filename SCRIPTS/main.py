from SCRIPTS import OPPD_utils

import os
import cv2

# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.


def addBndBox2All(data_path: str) -> dict:
    """
    This method is used to add bounding boxes to all images. It saves the images in a folder, as they are later
    needed for training.
    :param: data_path
    :return: a dictionary with an entry for each image, with the image id as the key and the image as an ndarray with
    bounding boxes as the element
    """
    # list of tuples: [folder, filename] for each image
    all_images = OPPD_utils.getImagesInFolder(data_path)
    all_images_bndbox = {}

    # iterate over all images
    for img in all_images:
        # generate the path to each image
        path_to_image = img[0]+"/"+img[1]
        # get image id and image with bounding boxes
        img_id, img_bndbox = OPPD_utils.addBndBoxes2Image(path_to_image)
        all_images_bndbox[img_id] = img_bndbox
        # create a unique title for each image: "[image_id].jpg"
        title = str(img_id)+".jpg"
        # path where the image is stored
        # TODO: at the moment the path leads always to ALOMY, needs to be plant specfic
        path = "/Users/laurenzohnemuller/PycharmProjects/PlantIdentification/images_full_bndbox"
        # generate image and store in path
        # Note: we don't need to store the images as actual images, the ndarrays will be enough (the matrix will be the
        # input for later models anyways)
        cv2.imwrite(os.path.join(path, title), img_bndbox)

    return all_images_bndbox

if __name__ == '__main__':

    data_path = "/Users/laurenzohnemuller/PycharmProjects/PlantIdentification/DATA"
    addBndBox2All(data_path)