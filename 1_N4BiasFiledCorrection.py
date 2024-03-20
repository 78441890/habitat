import SimpleITK as sitk
import warnings
import time
import os
from nipype.interfaces.ants import N4BiasFieldCorrection
import shutil
import numpy as np

def bias_field_correction_by_sitk(in_file, out_file):
    image = sitk.ReadImage(in_file)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    mask.SetSpacing(image.GetSpacing())
    mask.SetOrigin(image.GetOrigin())
    mask.SetDirection(image.GetDirection())
    image = sitk.Cast(image, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    output_image = corrector.Execute(image, mask)
    output_image = sitk.Cast(output_image, sitk.sitkInt16)
    sitk.WriteImage(output_image, out_file)


def correct_bias(in_file, out_file, image_type=sitk.sitkFloat64):
    """
    Corrects the bias using ANTs N4BiasFieldCorrection. If this fails, will then attempt to correct bias using SimpleITK
    :param in_file: nii文件的输入路径
    :param out_file: 校正后的文件保存路径名
    :return: 校正后的nii文件全路径名
    """

    # 使用N4BiasFieldCorrection校正MRI图像的偏置场
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file

    try:
        done = correct.run()
        return done.outputs.output_image
    except IOError:
        warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be found."
                                     "Will try using SimpleITK for bias field correction"
                                     " which will take much longer. To fix this problem, add N4BiasFieldCorrection"
                                     " to your PATH system variable. (example: EXPORT PATH=${PATH}:/path/to/ants/bin)"))

        input_image = sitk.ReadImage(in_file, image_type)
        output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
        sitk.WriteImage(output_image, out_file)

        return os.path.abspath(out_file)


if __name__ == "__main__":
    # in_file = "/home/liman/Projects/Trigeminal_nerve/intensity.nii"
    # out_file = "/home/liman/Projects/Trigeminal_nerve/intensity1.nii"
    # bias_field_correction_by_sitk(in_file, out_file)
    # correct_bias(in_file, out_file, image_type=sitk.sitkFloat64)

    root_dir = "/uiinas04/rd-rc/wangfang/11-zhongshan_wangcheng_HCC/"
    out_dir = "/data1/wangfang_data/zhongshan_HCC_wc/1_Data/1_N4ImagesAdd/"
    i = 31
    #for patient_id in np.sort(os.listdir(root_dir))[150]:
        #i += 1
    #patient_id = np.sort(os.listdir(root_dir))[31]
    for patient_id in ['ZS17018984','ZS17345900']:
        #print(i,' / ',len(os.listdir(root_dir)))
        in_file = os.path.join(root_dir,patient_id)
        out_file = os.path.join(out_dir, patient_id)
        if os.path.isdir(in_file):
            Series = os.listdir(in_file)
            for S in Series:
                Series_path = os.path.join(in_file,S)
                if os.path.isdir(Series_path):
                    last_dir = os.listdir(Series_path)
                    for L in last_dir:
                        last_dir_path = os.path.join(Series_path, L)
                        if os.path.isdir(last_dir_path):
                            file_names = np.sort(os.listdir(last_dir_path))
                            if(len(file_names)==2):
                                image_name = file_names[0]
                                roi_name = file_names[1]
                            if(len(file_names) == 3):
                                image_name = file_names[1]
                                roi_name = file_names[2]
                            print(str(i) + ":" + last_dir_path + " start!")
                            start = time.time()
                            InfileName = os.path.join(last_dir_path,image_name)
                            OutfileName = os.path.join(out_file,S,S+".nii.gz")
                            OutFolder = os.path.join(out_dir,patient_id,S)
                            try:
                                if not os.path.exists(OutFolder):
                                    os.makedirs(OutFolder)
                                    # outputName = os.path.join(outputfolder, s+'.nii.gz')
                                    bias_field_correction_by_sitk(InfileName, OutfileName)
                                    # roi copy
                                    roi_path = os.path.join(last_dir_path,roi_name)
                                    roi_save = os.path.join(out_file,S,S+"_mask.nii.gz")
                                    shutil.copy(roi_path, roi_save)
                                else:
                                    # outputName = os.path.join(outputfolder, s+'.nii.gz')
                                    bias_field_correction_by_sitk(InfileName, OutfileName)
                                    # roi copy
                                    roi_path = os.path.join(last_dir_path,roi_name)
                                    roi_save = os.path.join(out_file,S,S+"_mask.nii.gz")
                                    shutil.copy(roi_path, roi_save)
                                consume = time.time() - start
                                print(str(i) + ":" + in_file + " done!" + " time consuming:" + str(consume))
                            except:
                                print(str(i) + ":" + in_file + " faild!")
