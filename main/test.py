from Segmentation import *
from Collaboration import *
from Visualization import *
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-index", type=int, required=True)
    parser.add_argument("-csv_path", default="../img_detail/detail.csv")
    parser.add_argument("-data_path", default="../sim_data/euc_100.npy")
    parser.add_argument("-img_path", default="../imgs/")
    parser.add_argument("-blob_path", default="../img_blob/")
    parser.add_argument("-model_path", default="../final_model/ResNet_fold_0.pkl")

    opt = parser.parse_args()

    # segmentation
    ImageSegmentation = DTWImageSegmentation(opt.img_path, opt.blob_path)
    ImageSegmentation.segment_resized_G003_img(csv_path = opt.csv_path)

    # create collocative tensor
    create_similarity_dataset(csv_path = opt.csv_path, save_path = opt.data_path)

    # visualization
    data = np.load(opt.data_path)*get_mask(100)
    show = Grad_Cam_Main(index=opt.index,
                         data=data[opt.index], 
                         csv_path = opt.csv_path,
                         model_path= opt.model_path)
    bind_value = show()

if __name__ == "__main__":
    main()