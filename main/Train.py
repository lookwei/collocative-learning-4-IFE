from Segmentation import *
from Collaboration import *
from params import *
from Model_Train import *
import argparse

def main():
    ImageSegmentation = DTWImageSegmentation(args["img_path"], args["blob_path"])
    ImageSegmentation.segment_resized_G003_img(csv_path = args["img_info_path"])

    create_similarity_dataset(csv_path = args["img_info_path"], save_path = args["data_path"])

    train(args)

if __name__ == "__main__":
    main()