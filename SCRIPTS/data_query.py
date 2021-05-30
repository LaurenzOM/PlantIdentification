from SCRIPTS import OPPD_utils

import cv2

"""
This file contains methods, which can be used to query for specific data. 
"""

def find_image_by_ID(to_find_id, all_anno):
    path = "/Users/laurenzohnemuller/DATA_PlantIdentification/images_full/"
    # iterate through all plant annotations
    for img_id, row in enumerate(all_anno):

        if row["image_id"] == to_find_id:
            print(path + row["filename"])
            img = cv2.imread(path + row["filename"])
            id, output = OPPD_utils.addBndBoxes2Image(path + row["filename"])
            cv2.imshow("Image,", img)
            cv2.imshow("id", output)
            cv2.waitKey(0)


def find_Image_by_EPPO(to_find_eppo, all_anno):
    path = "/Users/laurenzohnemuller/DATA_PlantIdentification/images_full/"
    for img_id, row in enumerate(all_anno):

        for plant in row["plants"]:
            if plant["eppo"] == to_find_eppo:
                print(path + row["filename"])
                '''                
                img = cv2.imread(path + row["filename"])
                id, output = OPPD_utils.addBndBoxes2Image(path + row["filename"])
                cv2.imshow("Image,", img)
                cv2.imshow("id", output)
                cv2.waitKey(0)
                '''

def show_bndbox_of_ID(img_path, bndbox_id, color=(255, 0, 0), alpha=0.2, thickness=5):
    """ Add a polygon for a specific bounding box annotation associated with the image

      params:
          path_img: path to image
          color: color of the polygons
          alpha: transparency of polygons fill
          thickness: thickness of polygons lines

      returns:
          output: image with added polygons for each bounding box
      """
    path_anno = img_path.replace('.jpg', '.json')
    image = cv2.imread(img_path)
    anno = OPPD_utils.readJSONAnnotation(path_anno)
    output = image.copy()
    img_id = anno["image_id"]

    for plant in anno['plants']:
        if plant["bndbox_id"] == bndbox_id:
            bndbox = plant['bndbox']
            coor_bndbox = OPPD_utils.bndbox2polygon(bndbox)
            output = OPPD_utils.addPolygon2Image(output, coor_bndbox, color, alpha, thickness)

    return img_id, output


def find_small_BB(all_anno: list):
    """
    This method was used to find extremely small bounding boxes. Those were probably caused by an annotation error. Note
    that this method checks for small bounding boxes before resizing
    :param all_anno:
    :return:
    """

    path = "/Users/laurenzohnemuller/DATA_PlantIdentification/images_full/"

    filenames = {}
    # iterate over all iterations
    for anno in all_anno:
        for plant in anno["plants"]:
            bbox = plant["bndbox"]
            width = bbox["xmax"] - bbox["xmin"]
            height = bbox["ymax"] - bbox["ymin"]
            # width less than 9 pixels
            if width < 12 or height < 12:
                print("Width or height is less than 12 pixels")
                print(anno["filename"])
                print(plant["bndbox_id"])
                filenames[anno["filename"]] = plant["bndbox_id"]

    for file in filenames:
        id, output = show_bndbox_of_ID(path + file,
                                       filenames[file])
        cv2.imshow("file", output)
        cv2.waitKey(0)




def get_stats_about_eppo():
    """
    To get as much information as possible from one plant
    :return:
    """
    return None