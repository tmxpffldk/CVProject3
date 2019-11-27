import os
import numpy as np
from create_results_webpage import create_results_webpage
from helpers import get_image_paths
from student import build_vocabulary, get_bags_of_words, svm_classify


def projSceneRecBoW():
    '''
    For this project, you will need to report performance for three
    combinations of features / classifiers. We recommend that you code them in
    this order:
        - Bag of word features and linear SVM classifier
    The starter code is initialized to 'placeholder' just so that the starter
    code does not crash when run unmodified and you can get a preview of how
    results are presented.

    Interpreting your performance with 100 training examples per category:
     accuracy  =   0 -> Something is broken.
     accuracy ~= .07 -> Your performance is equal to chance.
                        Something is broken or you ran the starter code unchanged.
     accuracy ~= .20 -> Rough performance with tiny images and nearest
                        neighbor classifier. Performance goes up a few
                        percentage points with K-NN instead of 1-NN.
     accuracy ~= .20 -> Rough performance with tiny images and linear SVM
                        classifier. Although the accuracy is about the same as
                        nearest neighbor, the confusion matrix is very different.
     accuracy ~= .40 -> Rough performance with bag of word and nearest
                        neighbor classifier. Can reach .60 with K-NN and
                        different distance metrics.
     accuracy ~= .50 -> You've gotten things roughly correct with bag of
                        word and a linear SVM classifier.
     accuracy >= .70 -> You've also tuned your parameters well. E.g. number
                        of clusters, SVM regularization, number of patches
                        sampled when building vocabulary, size and step for
                        dense features.
     accuracy >= .80 -> You've added in spatial information somehow or you've
                        added additional, complementary image features. This
                        represents state of the art in Lazebnik et al 2006.
     accuracy >= .85 -> You've done extremely well. This is the state of the
                        art in the 2010 SUN database paper from fusing many
                        features. Don't trust this number unless you actually
                        measure many random splits.
     accuracy >= .90 -> You used modern deep features trained on much larger
                        image databases.
     accuracy >= .96 -> You can beat a human at this task. This isn't a
                        realistic number. Some accuracy calculation is broken
                        or your classifier is cheating and seeing the test
                        labels.
    '''

    #FEATURE = 'placeholder'
    FEATURE = 'bag of words'
    #FEATURE = 'SIFT'
    #FEATURE = 'GMM'
    #FEATURE = 'LBP'
    #FEATURE = 'SURF'

    CLASSIFIER = 'support vector machine'
    #CLASSIFIER = 'placeholder'

    data_path = './data/'

    categories = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
                  'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
                  'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

    abbr_categories = ['Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub',
                       'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst', 'Mnt', 'For']

    num_train_per_cat = 100

    print('Getting paths and labels for all train and test data.')
    train_image_paths, test_image_paths, train_labels, test_labels = \
        get_image_paths(data_path, categories, num_train_per_cat)

    print('Using %s representation for images.' % FEATURE)
    if FEATURE.lower() == 'bag of words' or FEATURE.lower() == 'sift'\
            or FEATURE.lower() == 'lbp' or FEATURE.lower() == 'gmm'\
            or FEATURE.lower() == 'surf':

        if not os.path.isfile('vocab.npy'):
            print('No existing visual word vocabulary found. Computing one from training images.')
            vocab_size = 200

            vocab = build_vocabulary(train_image_paths, vocab_size, FEATURE.lower())
            np.save('vocab.npy', vocab)
            print("success vocabulary generation")

        print("find vocabulary file")
        
        # 해당 코드에서는 임시로 image_feature를 뽑아내는 함수를 get_bags_of_words 라고 정의한다.
        train_image_feats = get_bags_of_words(train_image_paths, FEATURE.lower())
        test_image_feats = get_bags_of_words(test_image_paths, FEATURE.lower())

    elif FEATURE.lower() == 'placeholder':
        train_image_feats = []
        test_image_feats = []

    else:
        raise ValueError('Unknown feature type!')

    print('Using %s classifier to predict test set categories.' % CLASSIFIER)

    if CLASSIFIER.lower() == 'support vector machine':
        predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)

    elif CLASSIFIER.lower() == 'placeholder':
        random_permutation = np.random.permutation(len(test_labels))
        predicted_categories = [test_labels[i] for i in random_permutation]

    else:
        raise ValueError('Unknown classifier type')

    create_results_webpage(train_image_paths, \
                           test_image_paths, \
                           train_labels, \
                           test_labels, \
                           categories, \
                           abbr_categories, \
                           predicted_categories)

if __name__ == '__main__':
    projSceneRecBoW()
