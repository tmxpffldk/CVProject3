import numpy as np
from skimage import io, color
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn import mixture
from skimage.feature import hog, local_binary_pattern
from skimage.transform import resize
from scipy.stats import itemfreq
import cv2 as cv
import scipy.spatial.distance as distance

def build_vocabulary(image_paths, vocab_size, feature_name):
    #TODO: Implement this function!
    features = []
    image_size = 144
    z = 2
    for path in image_paths:
        if feature_name == 'bag of words':
            img = io.imread(path)
            img = resize(img, (image_size, image_size))
            long_boi = hog(img, orientations=9, cells_per_block=(2, 2), pixels_per_cell=(4, 4),
                           feature_vector=True, visualize=False)
            long_boi = np.array(long_boi)
            small_boi = long_boi.reshape(-1, z * z * 9)
            features.append(small_boi)

        elif feature_name == 'sift':
            img = cv.imread(path)
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            sift = cv.xfeatures2d.SIFT_create()
            kp, long_boi = sift.detectAndCompute(gray_img, None)
            long_boi = np.array(long_boi, dtype=float)
            features.append(long_boi)

        elif feature_name == 'gmm':
            img = io.imread(path)
            img = resize(img, (image_size, image_size))
            gmm = mixture.GaussianMixture(n_components=2)
            long_boi = gmm.fit(img)
            long_boi = long_boi.means_
            features.append(long_boi)

        elif feature_name == 'lbp':
            img = io.imread(path)
            img = resize(img, (image_size, image_size))
            radius = 3
            no_points = 8 * radius
            lbp = local_binary_pattern(img, no_points, radius, method="uniform")
            features.append(lbp)

        elif feature_name == 'surf':
            img = cv.imread(path)
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            surf = cv.xfeatures2d_SURF.create()
            kp, long_boi = surf.detectAndCompute(gray_img, None)
            long_boi = np.array(long_boi, dtype=float)
            features.append(long_boi)

    features = np.vstack(features)
    kmeans = KMeans(n_clusters=vocab_size, max_iter=100).fit(features)
    vocab = np.vstack(kmeans.cluster_centers_)

    return vocab

def get_bags_of_words(image_paths, feature_name):
    vocab = np.load('./vocab.npy')
    print('Loaded vocab from file.')

    #TODO: Implement this function!
    image_features = []
    z = 2
    image_size = 144
    for path in image_paths:
        if feature_name == 'bag of words':
            img = io.imread(path)
            img = resize(img, (image_size, image_size))
            long_boi = hog(img, orientations=9, cells_per_block=(2, 2), pixels_per_cell=(4, 4),
                           feature_vector=True, visualize=False)
            long_boi = np.array(long_boi).reshape(-1, z*z*9)

        elif feature_name == 'sift':
            img = cv.imread(path)
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            sift = cv.xfeatures2d.SIFT_create()
            kp, long_boi = sift.detectAndCompute(gray_img, None)
            long_boi = np.array(long_boi, dtype=float)

        elif feature_name == 'gmm':
            img = io.imread(path)
            img = resize(img, (image_size, image_size))
            gmm = mixture.GaussianMixture(n_components=2)
            long_boi = gmm.fit(img)
            long_boi = long_boi.means_

        elif feature_name == 'lbp':
            img = io.imread(path)
            img = resize(img, (image_size, image_size))
            radius = 3
            no_points = 8 * radius
            lbp = local_binary_pattern(img, no_points, radius, method="uniform")
            long_boi = lbp

        elif feature_name == 'surf':
            img = cv.imread(path)
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            surf = cv.xfeatures2d_SURF.create()
            kp, long_boi = surf.detectAndCompute(gray_img, None)
            long_boi = np.array(long_boi, dtype=float)

        dist = distance.cdist(long_boi, vocab)
        closet = np.argsort(dist, axis=1)[:,0]
        hist = np.zeros(len(vocab))
        idx, counts = np.unique(closet, return_counts=True)
        hist[idx] += counts
        hist_norm = hist / np.linalg.norm(hist)
        image_features.append(hist_norm)
    return image_features

def svm_classify(train_image_feats, train_labels, test_image_feats):
    # TODO: Implement this function!
    SVC = LinearSVC(random_state=0, tol=1e-5)
    SVC.fit(train_image_feats, train_labels)

    return SVC.predict(test_image_feats)
