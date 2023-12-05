import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from pathlib import Path
import cv2 as cv
from feat import Detector
from feat.utils import FEAT_EMOTION_COLUMNS


# ----------------------------------#

def process_file_name(s):
    return s.split('\\')[-1]


def process_file_name2(s):
    return s.split('/')[-1]


def main():
    # -----------------READING IMAGES FROM DISK-----------------#

    # Some steps to fix the paths for saving the csv and images
    path = Path(os.getcwd()).parent
    DIR_PATH = str(Path(__file__).parent.parent.absolute()) + r"\\"

    emo_folders = Path(DIR_PATH + 'data/DiffusionFER/DiffusionEmotion_S/cropped').glob('*')
    emo_folder_paths = [str(i) for i in emo_folders]

    # List containing paths for all images across all emotions in 'cropped'
    img_path_list_combined = []

    for e in emo_folder_paths:
        img_emo_folder = Path(e).glob('*.png')
        img_emo_paths = [str(i) for i in img_emo_folder]
        img_path_list_combined.extend(img_emo_paths)

    # -----------------DETECTING FACIAL ACTION UNITS (AND OTHER STUFF) FROM IMAGES -----------------#

    detector = Detector(device="cuda")
    mixed_prediction = detector.detect_image(img_path_list_combined)

    # -----------------CREATING THE AU DATAFRAME-----------------#

    ext_columns = mixed_prediction.aus.columns.tolist() + mixed_prediction.emotions.columns.tolist() + ['input']

    # Column wise slice & dropping rows with NaN values (possibly cases where no face is detected or something else
    # got misclassified as a face)
    sliced_df = mixed_prediction[ext_columns].dropna()

    # -----------------CREATING THE AU DATAFRAME-----------------#

    sliced_df['input'] = sliced_df.inputs.apply(process_file_name)

    # -----------------READING THE DATASHEET CSV-----------------#

    df = pd.read_csv(DIR_PATH + 'data/DiffusionFER/DiffusionEmotion_S/dataset_sheet.csv')
    df['subDirectory_filePath'] = df['subDirectory_filePath'].apply(process_file_name2)
    df.rename(columns={'subDirectory_filePath': 'input'}, inplace=True)

    # -----------------MERGING THE DATAFRAMES WITH RESPECT TO INPUT(IMAGES)-----------------#

    df_final = pd.merge(sliced_df, df, on='input', how='left')

    # Saving the df to disk
    df_final.to_csv(DIR_PATH + 'data/extracted_df.csv', index=False)


if __name__ == '__main__':
    main()
