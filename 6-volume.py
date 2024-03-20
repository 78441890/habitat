import os
import nibabel as nib
import numpy as np
import csv

def calculate_volumes(patient_id, image_path, total_roi_path, label_roi_path):
    image_file_path = os.path.join(image_path, patient_id + '.nii.gz')
    print(image_file_path)
    total_roi_file_path = os.path.join(total_roi_path, patient_id + '.nii.gz')
    print(total_roi_file_path)
    label_roi_file_path = os.path.join(label_roi_path, patient_id + '_otsu1.nii.gz')
    print(label_roi_file_path)

    image = nib.load(image_file_path).get_fdata()
    total_roi = nib.load(total_roi_file_path).get_fdata()
    label_roi = nib.load(label_roi_file_path).get_fdata()

    total_roi_volumes = np.count_nonzero(total_roi)
    print(total_roi_volumes)

    labels = np.unique(label_roi)
    volumes_dict = {'patient_id': patient_id, 'total_roi_volume': total_roi_volumes}

    for label in labels:
        if label == 0:
            continue
        label_volume = np.count_nonzero(label_roi == label)
        print(label)
        print(label_volume)
        volume_key = 'label{}_volume'.format(label)
        proportion_key = 'label{}_proportion'.format(label)
        volumes_dict[volume_key] = label_volume
        volumes_dict[proportion_key] = label_volume / total_roi_volumes

    return volumes_dict
        
def save_to_csv(volumes_dict_list, csv_path):
    keys = volumes_dict_list[0].keys()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(volumes_dict_list)


def main(image_folder_path, total_roi_folder_path, label_roi_folder_path, csv_path):
    volumes_dict_list = []
    filenames = os.listdir(image_folder_path)
    for filename in filenames:
        patient_id = filename.split('.')[0]
        volumes_dict = calculate_volumes(patient_id, image_folder_path, total_roi_folder_path, label_roi_folder_path)
        volumes_dict_list.append(volumes_dict)
    save_to_csv(volumes_dict_list, csv_path)

image_folder_path = "/home/yuwei/code/example_data/T2/"
total_roi_folder_path ="/home/yuwei/code/example_data/mask/"
label_roi_folder_path = "/home/yuwei/code/example_data/otsu_1009/"
csv_path = "/home/yuwei/code/example_data/result_1009.csv"

main(image_folder_path, total_roi_folder_path, label_roi_folder_path,csv_path)