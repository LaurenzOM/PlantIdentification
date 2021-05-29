import matplotlib.pyplot as plt

"""
This file is used to represent the data different ways.
"""


def class_distribution(all_anno: list) -> None:
    """
    This method shows the distribution of categories/plants. Note that the plant objects are counted here.
    :param all_anno: a list of all annotations
    :return: a plot that shows the distribution of plant objects
    """
    plant_names = ("ALOMY ", "ANGAR ", "APESV ", "ARTVU ", "AVEFA ", "BROST ", "BRSNN ", "CAPBP ", "CENCY ", "CHEAL ",
                   "CHYSE ", "CIRAR ", "CONAR ","EPHHE ","EPHPE ","EROCI ", "FUMOF ", "PPPDD ", "PPPMM ")
    counts = [0]*len(plant_names)
    for anno in all_anno:
        for plant in anno["plants"]:
            for plant_idx in range(len(plant_names)):
                if plant["eppo"] == plant_names[plant_idx]:
                    counts[plant_idx] += 1

    plt.bar(plant_names, counts)
    plt.xticks(fontsize=7, rotation=90)
    plt.ylabel("Number of Plant Objects")
    plt.xlabel("Class of Plant")
    plt.show()


def plant_distribution(set: list, data_set_type: str) -> None:
    """
    This method is used to describe the given set in terms of the total amount of plant objects, the mean number of plant
    objects, the number of images without any plant and the ratio of how many images are without any plants.
    :param set: is a set of annotations
    :param data_set_type: is a flag to describe which set. Possible values: "test", "train", "val"
    :return: None, just some print statements containing that information
    """
    # to count total amount of plants
    total_amt_plants = 0
    # to count how many images have 0 plants
    count_zeros = 0

    for anno in set:
        plants = anno["plants"]
        total_amt_plants += len(plants)

        if len(plants) == 0:
            count_zeros += 1

    print("Total amount of plant objects in set:", data_set_type, ", is:", total_amt_plants)
    print("Mean number of plant objects in image", total_amt_plants/len(set))
    print("Number of images without any plants", count_zeros, "and in ratio to the number of images",len(set)/count_zeros)
    print("--------------------------------------------------")

