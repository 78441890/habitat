import os
from PIL import ImageColor
import random
import numpy as np
import cupy as cp
import nibabel as nib
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn import metrics

def plot_cluster(x, labels, n_clusters, filename):
    # colors = ['r', 'g']
    # colors = ['r', 'g', 'b',]
    # random.seed(20)
    # colors = ImageColor.getrgb('#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'pink', 'black', 'gold']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(n_clusters):
        ax.scatter(x[0, labels == i], x[1, labels == i], x[2, labels == i], c=colors[i],
                   label='Cluster {}'.format(i + 1), cmap='viridis')

    ax.set_xlabel('DWI', fontsize=8)
    ax.set_ylabel('T2', fontsize=8)
    ax.set_zlabel('Arterial', fontsize=8)
    ax.legend(fontsize=8)

    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.zaxis.set_tick_params(labelsize=8)

    plt.savefig(filename, dpi=300)
    plt.close()
    print(filename + "k-means picture finshed")


def perform_kmeans(all_data, test_all_data, output_folder):
    distortions = [];silhouette_scores = []; CH_scores = [];DB_score = []
    K = range(2, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(all_data)
        #kmeanModel2 = KMeans(n_clusters=k).fit_predict(all_data)
        # save result
        distortions.append(sum(np.min(cdist(all_data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / np.array(all_data).shape[0])
        silhouette_scores.append(silhouette_score(all_data, kmeanModel))
        CH_scores.append(metrics.calinski_harabasz_score(all_data, kmeanModel))
        DB_score.append(metrics.davies_bouldin_score(all_data, kmeanModel))
        print(str(k))

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('n_clusters')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal n_clusters')
    plt.savefig(os.path.join(output_folder, 'elbow.png'))
    plt.close()

    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('n_clusters')
    plt.ylabel('silhouette_scores')
    plt.title('The Silhouette Method showing the optimal n_clusters')
    plt.savefig(os.path.join(output_folder, 'Silhouette.png'))
    plt.close()

    plt.plot(K, CH_scores, 'bx-')
    plt.xlabel('n_clusters')
    plt.ylabel('CH_scores')
    plt.title('The CH Method showing the optimal n_clusters')
    plt.savefig(os.path.join(output_folder, 'CH_scores.png'))
    plt.close()

    plt.plot(K, DB_score, 'bx-')
    plt.xlabel('n_clusters')
    plt.ylabel('DB_scores')
    plt.title('The DB Method showing the optimal n_clusters')
    plt.savefig(os.path.join(output_folder, 'DB_scores.png'))
    plt.close()
    # n_clusters = np.diff(distortions, 2).argmax() + 2

    # line confusd
    print(distortions)
    print(silhouette_scores)
    print(CH_scores)
    print(DB_score)

    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_data)
    labels = kmeans.labels_

    # apply to testing data
    test_labels = kmeans.predict(test_all_data)

    plot_cluster(np.array(all_data).T, labels, n_clusters,
                 os.path.join(output_folder, 'kmeans_cluster_all_patients.png'))
    print("Elbow picture finshed")
    return labels, n_clusters,test_labels


def process_image_data(path_to_image_folder1, path_to_image_folder2, path_to_image_folder3, roi_folder, output_folder):
    filenames = os.listdir(path_to_image_folder1)
    all_data = []
    rois = []
    for filename in filenames:
        if filename.endswith(".nii.gz"):
            img1 = nib.load(os.path.join(path_to_image_folder1, filename))
            img2 = nib.load(os.path.join(path_to_image_folder2, filename))
            img3 = nib.load(os.path.join(path_to_image_folder3, filename))
            roi = nib.load(os.path.join(roi_folder, filename))
            img1_data = img1.get_fdata()
            img2_data = img2.get_fdata()
            img3_data = img3.get_fdata()
            roi_data = roi.get_fdata()
            roi_mask = roi_data != 0
            x = img1_data[roi_mask]
            y = img2_data[roi_mask]
            z = img3_data[roi_mask]
            rois.append({'roi_data': roi_data, 'roi_mask': roi_mask, 'affine': roi.affine})
            all_data.extend(np.column_stack((x, y, z)))

    labels, n_clusters = perform_kmeans(all_data, output_folder)

    start = 0
    for i, filename in enumerate(filenames):
        if filename.endswith(".nii.gz"):
            roi_data = rois[i]['roi_data']
            roi_mask = rois[i]['roi_mask']
            affine = rois[i]['affine']
            roi_kmeans = roi_data.copy()
            end = start + np.count_nonzero(roi_mask)
            roi_kmeans[roi_mask] = labels[start:end] + 1
            cluster_nii = nib.Nifti1Image(roi_kmeans, affine=affine)
            nib.save(cluster_nii, os.path.join(output_folder, filename.replace(".nii.gz", "_kmeans.nii.gz")))
            plot_cluster(np.array(all_data)[start:end].T, labels[start:end], n_clusters,
                         os.path.join(output_folder, filename.replace(".nii.gz", "_kmeans.png")))
            start = end


def process_image_data2(path_to_image_folder_regis, path_to_image_folder_Arterial, output_folder,test_ratio):
    patients = np.sort(os.listdir(path_to_image_folder_regis))
    # data split
    random.seed(100)
    train_patients, test_patients = train_test_split(patients, test_size=test_ratio)
    print("train data sample: " + str(len(train_patients)))
    print("test data sample: " + str(len(test_patients)))
    # kmeans data
    all_data = []; test_all_data = cp.asarray([])
    rois = []; test_rois = []
    i = 0
    for patient in np.sort(train_patients):
        i += 1

        img1 = nib.load(os.path.join(path_to_image_folder_regis, patient, "DWI", "DWI.nii.gz"))
        img2 = nib.load(os.path.join(path_to_image_folder_regis, patient, "T2", "T2.nii.gz"))
        img3 = nib.load(os.path.join(path_to_image_folder_regis, patient, "Arterial", "Arterial.nii.gz"))
        #roi = nib.load(os.path.join(path_to_image_folder_Arterial, patient, "Arterial", "Arterial_mask.nii.gz"))
        roi = nib.load(os.path.join(path_to_image_folder_Arterial, patient, "Arterial", "Arterial.nii.gz"))
        img1_data = img1.get_fdata()
        img2_data = img2.get_fdata()
        img3_data = img3.get_fdata()
        roi_data = roi.get_fdata()
        roi_mask = roi_data != 0
        x = img1_data[roi_mask]
        y = img2_data[roi_mask]
        z = img3_data[roi_mask]
        rois.append({'roi_data': roi_data, 'roi_mask': roi_mask, 'affine': roi.affine})
        # all_data.extend(cp.column_stack((cp.asarray(x), cp.asarray(y), cp.asarray(z))))
        temp = cp.column_stack((cp.asarray(x), cp.asarray(y), cp.asarray(z)))
        all_data = cp.append(all_data, temp)

        #all_data.extend(np.column_stack((x, z)))
        # if i>=200:
        #     print(i)
        print(str(i) + ": train data: "+ patient + " read success")

    flag = 0
    for patient in np.sort(test_patients):
        flag += 1
        img1 = nib.load(os.path.join(path_to_image_folder_regis, patient, "DWI", "DWI.nii.gz"))
        img2 = nib.load(os.path.join(path_to_image_folder_regis, patient, "T2", "T2.nii.gz"))
        img3 = nib.load(os.path.join(path_to_image_folder_Arterial, patient, "Arterial", "Arterial.nii.gz"))
        #roi = nib.load(os.path.join(path_to_image_folder_Arterial, patient, "Arterial", "Arterial_mask.nii.gz"))
        roi = nib.load(os.path.join(path_to_image_folder_Arterial, patient, "Arterial", "Arterial.nii.gz"))
        img1_data = img1.get_fdata()
        img2_data = img2.get_fdata()
        img3_data = img3.get_fdata()
        roi_data = roi.get_fdata()
        roi_mask = roi_data != 0
        x = img1_data[roi_mask]
        y = img2_data[roi_mask]
        z = img3_data[roi_mask]
        test_rois.append({'roi_data': roi_data, 'roi_mask': roi_mask, 'affine': roi.affine})
        # test_all_data.extend(cp.column_stack((cp.asarray(x), cp.asarray(y), cp.asarray(z))))
        temp = cp.column_stack((cp.asarray(x), cp.asarray(y), cp.asarray(z)))
        test_all_data = cp.append(test_all_data, temp)

        print(str(flag) + ": test data: " + patient + " read success")

    print("data read finshed")
    labels, n_clusters,test_labels  = perform_kmeans(all_data, test_all_data, output_folder)

    start = 0
    for i, filename in enumerate(np.sort(train_patients)):
        roi_data = rois[i]['roi_data']
        roi_mask = rois[i]['roi_mask']
        affine = rois[i]['affine']
        roi_kmeans = roi_data.copy()
        end = start + np.count_nonzero(roi_mask)
        roi_kmeans[roi_mask] = labels[start:end] + 1
        cluster_nii = nib.Nifti1Image(roi_kmeans, affine=affine)
        nib.save(cluster_nii, os.path.join(output_folder, filename.replace(".nii.gz", "_kmeans_train.nii.gz")))

        plot_cluster(np.array(all_data)[start:end].T, labels[start:end], n_clusters,
                     os.path.join(output_folder, filename.replace(".nii.gz", "_kmeans_train.png")))
        start = end

    # test data save
    start = 0
    for i, filename in enumerate(np.sort(test_patients)):
        test_roi_data = test_rois[i]['roi_data']
        test_roi_mask = test_rois[i]['roi_mask']
        test_affine = test_rois[i]['affine']
        test_roi_kmeans = test_roi_data.copy()
        end = start + np.count_nonzero(test_roi_mask)
        test_roi_kmeans[test_roi_mask] = test_labels[start:end] + 1
        test_cluster_nii = nib.Nifti1Image(test_roi_kmeans, affine=test_affine)
        nib.save(test_cluster_nii, os.path.join(output_folder, filename.replace(".nii.gz", "_kmeans_test.nii.gz")))

        plot_cluster(np.array(test_all_data)[start:end].T, test_labels[start:end], n_clusters,
                     os.path.join(output_folder, filename.replace(".nii.gz", "_kmeans_test.png")))
        start = end
    return train_patients, test_patients


if __name__ == "__main__":
    path_to_image_folder_regis = "/data1/wangfang_data/zhongshan_HCC_wc/1_Data/3_Normalization/"
    path_to_image_folder_Arterial = "/data1/wangfang_data/zhongshan_HCC_wc/1_Data/6_dilation5mm/"
    output_folder = "/data1/wangfang_data/zhongshan_HCC_wc/1_Data/4_K-means_auto_5mmdilaton/"
    test_ratio = 0.3

    train_patients, test_patients = process_image_data2(path_to_image_folder_regis, path_to_image_folder_Arterial, output_folder,test_ratio)
