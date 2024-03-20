from __future__ import print_function
import os
import csv
import time
import argparse
import numpy as np
import multiprocessing
import copy

import md.image3d.python.image3d_io as cio
from md_segmentation3d.vseg import autoseg_volume, autoseg_load_model
from md_segmentation3d.utils.imlist_tools import MedImagePattern
from md_segmentation3d.utils.vseg_helpers import read_test_txt, read_test_csv, read_test_folder

message_queue = multiprocessing.Queue()

# 理论上面multiprocessing是线程安全的，可以不用加锁，保险起见还是加了
queue_lock = multiprocessing.Lock()
csv_lock = multiprocessing.Lock()
model_lock = multiprocessing.Lock()


def worker(gpu_id, model_folder, output_folder, seg_name='seg.mha',
           save_image=True, save_single_prob=False):
    model_lock.acquire()
    model = autoseg_load_model(model_folder, gpu_id)
    model_lock.release()

    while (1):
        queue_lock.acquire()
        if message_queue.empty():
            queue_lock.release()
            return
        else:
            message = message_queue.get()
            queue_lock.release()

        message_split = message.split(',')
        modality_num = len(message_split) - 2

        index = message_split[0]
        casename = message_split[-1]

        print("processing {}: {}".format(index, message_split[1]))

        begin = time.time()
        images = []
        for image_path in message_split[1:modality_num - 2]:
            image = cio.read_image(image_path, dtype=np.float32)
            images.append(image)
        read_time = time.time() - begin

        begin = time.time()
        try:
            mask, prob_map = autoseg_volume(images, model)
        except Exception as e:
            print('fails to segment volume: ', message_split[1], ', {}'.format(e))
            continue
        test_time = time.time() - begin

        name = MedImagePattern()
        if name.test(output_folder):
            begin = time.time()
            out_folder, _ = os.path.split(output_folder)
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder)
            cio.write_image(mask, output_folder, compression=True)
            output_time = time.time() - begin
        else:
            out_folder = os.path.join(output_folder, casename)
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder)

            begin = time.time()
            if save_image:
                if len(images) == 1:
                    ct_path = os.path.join(out_folder, 'org.mha')
                    cio.write_image(images[0], ct_path)
                else:
                    for num in range(len(images)):
                        ct_path = os.path.join(out_folder, 'org{}'.format(num + 1) + '.mha')
                        cio.write_image(images[num], ct_path)

            seg_path = os.path.join(out_folder, seg_name)
            cio.write_image(mask, seg_path, compression=True)

            if save_single_prob and prob_map:
                prob_path = os.path.join(out_folder, 'prob.mhd')
                cio.write_image(prob_map, prob_path, compression=True)
            output_time = time.time() - begin

        total_time = read_time + test_time + output_time
        print('segmentation {}: read: {:.2f} s, test: {:.2f} s, write: {:.2f} s'.format(
            index, read_time, test_time, output_time))


def batch_segmentation(input_path, model_folder, output_folder, seg_name='seg.mha', gpus_id=[0], save_image=True,
                       save_single_prob=False):
    """ volumetric image segmentation engine
    :param input_path:          a path of text file, a single image file
                                or a root dir with all image files
    :param model_folder:        path of trained model
    :param output_folder:       path of out folder
    :param gpu_id:              which gpu to use, by default, 0
    :param save_image           whether to save original image
    :return: None
    """

    # determine testing cases
    suffix = ['.mhd', '.nii', '.hdr', '.nii.gz', '.mha', '.image3d']
    name = MedImagePattern()
    if os.path.isfile(input_path):
        if name.test(input_path):
            # test just one case (single-modality)
            im_name = os.path.basename(input_path)
            for suf in suffix:
                idx = im_name.find(suf)
                if idx != -1:
                    im_name = im_name[:idx]
                    break
            file_list = [[input_path]]
            case_list = [im_name]
        else:
            # test image files in the text (single-modality) or csv (multi-modality) file
            if input_path.endswith('txt'):
                file_list, case_list = read_test_txt(input_path)
            elif input_path.endswith('csv'):
                file_list, case_list = read_test_csv(input_path)
            else:
                raise ValueError('image test_list must either be a txt file or a csv file')
    elif os.path.isdir(input_path):
        # test all image file in input folder (single-modality)
        file_list, case_list = read_test_folder(input_path)
    else:
        raise ValueError('Input path do not exist!')

    for i, files in enumerate(file_list):
        modality_num = len(files)
        message = str(i)
        for j in range(modality_num):
            message = message + ",{}".format(files[j])

        message = message + ",{}".format(case_list[i])
        message_queue.put(message)

    process_list = list()
    for id in gpus_id:
        p = multiprocessing.Process(target=worker,
                                    args=(id, model_folder, output_folder, seg_name, save_image, save_single_prob))
        process_list.append(p)
        p.start()


def main():
    from argparse import RawTextHelpFormatter

    long_description = 'UII Segmentation3d Batch Testing Engine\n\n' \
                       'It supports multiple kinds of input:\n' \
                       '1. Image list txt file\n' \
                       '2. Single image file\n' \
                       '3. A folder that contains all testing images\n'

    parser = argparse.ArgumentParser(description=long_description,
                                     formatter_class=RawTextHelpFormatter)

    # input image path
    input_path = '/data1/wangfang_data/zhongshan_HCC_wc/1_Data/3_Normalization/'
    i = 0
    for p in os.listdir(input_path):
        i += 1
        image_path = os.path.join(input_path,p,'Arterial','Arterial.nii.gz')
        parser.add_argument('-i', '--input', default=image_path,type=str, help='input folder/file for intensity images')
        parser.add_argument('-m', '--model', default='/home/wangfang/PythonRelated/MR_Liver_C++/',type=str, help='model root folder')
        parser.add_argument('-o', '--output', default='/data1/wangfang_data/zhongshan_HCC_wc/1_Data/5-liverSegmentation/', type=str, help='output folder for segmentation')
        parser.add_argument('-n', '--seg_name', default='_liver.nii.gz', help='the name of the segmentation result to be saved')
        parser.add_argument('-g', '--gpus_id', default="0,1,2,3", help='the gpu id to run model')
        parser.add_argument('--save_image', default=False, help='whether to save original image', action="store_true")
        parser.add_argument('--save_single_prob', default=False, help='whether to save single prob map', action="store_true")
        args = parser.parse_args()

        batch_segmentation(args.input, args.model, args.output, args.seg_name,
                           [int(gpu) for gpu in args.gpus_id.split(',')], args.save_image,
                           args.save_single_prob)
        print(i+ " | "+len(os.listdir(input_path))+ " liver segmentation success")


if __name__ == '__main__':
    main()
