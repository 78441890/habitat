import os
import numpy as np
import SimpleITK as sitk
import md.image3d.python.image3d_io as cio
import shutil
import copy

def intersectionROI(liver_path,dilation_path):
    for p in np.sort(os.listdir(liver_path)):
        image_path_liver = os.path.join(liver_path,p,"liver.nii.gz")
        tmp_path = os.path.join(dilation_path,p,"Arterial")
        image_path_dilation = os.path.join(tmp_path,os.listdir(tmp_path)[0])

        image_liver = sitk.ReadImage(image_path_liver)
        image_dilation = sitk.ReadImage(image_path_dilation)
        # Convert to numpy array and get ROI mask
        image_liver_array = sitk.GetArrayFromImage(image_liver)
        image_dilation_array = sitk.GetArrayFromImage(image_dilation)
        image_write_intersection = image_dilation_array * image_liver_array
        image_write_intersection[image_write_intersection>1] = 1

        imageread = cio.read_image(image_path_dilation)
        imageread.from_numpy(image_write_intersection, dtype = np.uint8)

        # write out
        #OutFolder = image_path_dilation.replace(".nii.gz","_sure.nii.gz")
        OutFolder = os.path.join(tmp_path,"Arterial.nii.gz")
        cio.write_image(imageread, OutFolder)
        print(p+" instersection success")

def HabitatROISave(dilation_roi_path,dilation_habitat_roi_path,image_path,save_path):
    flag = 0; Lenth = len(os.listdir(image_path))
    for p in np.sort(os.listdir(image_path)):
        flag += 1
        # dilation_habitat_roi
        dilation_path = os.path.join(dilation_roi_path, p, "Arterial", "Arterial.nii.gz")
        # habitat ROI
        image_path_habitat = os.path.join(dilation_habitat_roi_path, p + ".nii")
        image_habitat = sitk.ReadImage(image_path_habitat)
        image_habitat_array = sitk.GetArrayFromImage(image_habitat)
        UniValue = np.unique(image_habitat_array)
        # UniValue = set(image_habitat_array)
        print("dilation_habitat ROI read finished")
        for value in np.sort(UniValue)[1:]:
            save_habitat_image = copy.deepcopy(image_habitat_array)
            save_habitat_image[save_habitat_image != int(value)] = 0
            save_habitat_image[save_habitat_image > 0] = 1
            #save_habitat_image = save_habitat_image.astype(int)
            # save ROI
            tmp_path1 = os.path.join(image_path, p)
            for s in os.listdir(tmp_path1):
                # image
                imaage_path_sure = os.path.join(image_path, p, s, s + ".nii.gz")
                # image shutil
                orin_image_p_path = os.path.join(save_path, p)
                orin_image_save_path = os.path.join(orin_image_p_path, s)
                if not os.path.exists(orin_image_p_path):
                    os.mkdir(orin_image_p_path)
                if not os.path.exists(orin_image_save_path):
                    os.mkdir(orin_image_save_path)
                shutil.copy(imaage_path_sure, os.path.join(orin_image_save_path, s + ".nii.gz"))
                shutil.copy(dilation_path, os.path.join(orin_image_save_path, s + "_dilation.nii.gz"))
                # habitat roi save
                dilation_imageread = cio.read_image(dilation_path)
                dilation_imageread.from_numpy(save_habitat_image, dtype = np.uint8)

                OutFolder = os.path.join(orin_image_save_path, s + "_dilation_" + str(value) + ".nii.gz")
                cio.write_image(dilation_imageread, OutFolder)
                print(str(flag)+"/"+str(Lenth)+": "+ p +"/"+s+" ROI=",str(int(value))+" dealt finished!")

if __name__ == '__main__':
    # liver_path = "/data1/wangfang_data/zhongshan_HCC_wc/1_Data/5-liverSegmentation/"
    # dilation_path = "/data1/wangfang_data/zhongshan_HCC_wc/1_Data/6_dilation5mm/"
    # dilation10mm_path = "/data1/wangfang_data/zhongshan_HCC_wc/1_Data/7_dilation10mm/"
    # intersectionROI(liver_path,dilation_path)
    # intersectionROI(liver_path, dilation10mm_path)
    dilation_roi_path = "/data1/wangfang_data/zhongshan_HCC_wc/1_Data/6_dilation5mm/"
    dilation_habitat_roi_path = "/data1/wangfang_data/zhongshan_HCC_wc/1_Data/4_K-means_auto_5mmdilaton/"
    image_path = "/data1/wangfang_data/zhongshan_HCC_wc/1_Data/3_Normalization/"
    save_path = "/data1/wangfang_data/zhongshan_HCC_wc/1_Data/8_dilation_habitat_5mm/"
    HabitatROISave(dilation_roi_path,dilation_habitat_roi_path,image_path,save_path)
    #
    # dilation_roi_path = "/data1/wangfang_data/zhongshan_HCC_wc/1_Data/7_dilation10mm/"
    # dilation_habitat_roi_path = "/data1/wangfang_data/zhongshan_HCC_wc/1_Data/4_K-means_auto_10mmdilaton/"
    # image_path = "/data1/wangfang_data/zhongshan_HCC_wc/1_Data/3_Normalization/"
    # save_path = "/data1/wangfang_data/zhongshan_HCC_wc/1_Data/8_dilation_habitat_10mm/"
    # HabitatROISave(dilation_roi_path,dilation_habitat_roi_path,image_path,save_path)