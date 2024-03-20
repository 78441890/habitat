import os
import SimpleITK as sitk
import numpy as np
import md.image3d.python.image3d_io as cio


def normalize(image_array, min_p=0.01, max_p=0.99):
    """
    normalize image, rescaled to [-1, 1], without background voxels in statistic analysis
    """
    imgArray = np.float32(image_array)

    imgPixel = imgArray[imgArray >= 0]
    imgPixel.sort()
    index = int(round(len(imgPixel) - 1) * min_p + 0.5)
    if index < 0:
        index = 0
    if index > (len(imgPixel) - 1):
        index = len(imgPixel) - 1
    value_min = imgPixel[index]

    indexmax = int(round(len(imgPixel) - 1) * max_p + 0.5)
    if indexmax < 0:
        indexmax = 0
    if indexmax > (len(imgPixel) - 1):
        indexmax = len(imgPixel) - 1
    value_max = imgPixel[indexmax]

    # mean = (value_max + value_min) / 2.0
    # stddev = (value_max - value_min) / 2.0
    #
    # mean = np.mean(imgPixel[index:indexmax])
    # stddev = np.std(imgPixel[index:indexmax])
    #imgArray_process = imgArray / np.abs(value_max)
    imgArray_process = (imgArray -value_min)/ (value_max-value_min)
    imgArray_process[imgArray_process < -1] = -1.0
    imgArray_process[imgArray_process > 1] = 1.0

    return imgArray_process


if __name__ == '__main__':
    root_dir = "/data1/wangfang_data/zhongshan_HCC_wc/1_Data/2_RegistrationAdd/"
    save_dir = "/data1/wangfang_data/zhongshan_HCC_wc/1_Data/3_Normalization/"
    flag = 0

    problemID = []
    for patient in np.sort(os.listdir(root_dir)):
    #for patient in ['ZS12205588','ZS17018984','ZS17345900']:
        flag += 1
        p_path = os.path.join(root_dir,patient)
        try:
            for s in os.listdir(p_path):
                s_path = os.path.join(p_path,s)
                # image process
                image_path = os.path.join(s_path,s+".nii.gz")
                # load image
                image = sitk.ReadImage(image_path)
                # Convert to numpy array and get ROI mask
                image_array = sitk.GetArrayFromImage(image)
                image_process = normalize(image_array)
                # write our
                OutFolder = os.path.join(save_dir,patient,s)
                if not os.path.exists(OutFolder):
                    os.makedirs(OutFolder)

                imageread = cio.read_image(image_path)
                imageread.from_numpy(image_process)
                cio.write_image(imageread, OutFolder + "/" + s + ".nii.gz")
                print(str(flag) + ": " + s_path + " normalization sucessful")
        except:
            problemID.append(patient)
    print(problemID)




