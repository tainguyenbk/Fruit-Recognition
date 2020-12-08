# -----------------------------------
# TRAINING OUR MODEL
# -----------------------------------
import h5py
import numpy as np
import os
import glob
from global_features import fixed_size, fd_haralick, fd_histogram, fd_hu_moments, MinMaxScaler
import cv2
import warnings
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import joblib
import random
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# --------------------
# tunable-parameters
# --------------------
num_trees = 20
test_size = 0.25
seed = 9
train_path = "dataset/new_train"
test_path = "dataset/new_test"
h5_data = "output/data.h5"
h5_labels = "output/labels.h5"
scoring = "accuracy"
images_per_class = 80
# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()

# get the training labels
test_labels = os.listdir(test_path)

# sort the training labels
test_labels.sort()

# if not os.path.exists(test_path):
#     os.makedirs(test_path)

# create all the machine learning models
# models = []
# models.append(('LR', LogisticRegression(random_state=seed)))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier(random_state=seed)))
# models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(random_state=seed)))

# # variables to hold the results and names
# results = []
# names = []

# import the feature vector and trained labels
h5f_data = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)

global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of the feature vector and labels
print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))
print("[STATUS] training started...")
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features), np.array(global_labels), test_size=1/320, random_state=seed)
# clf = GaussianNB()
clf1 = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
clf2 = KNeighborsClassifier()
clf3 = GaussianNB()
clf4 = LinearDiscriminantAnalysis()
clf5 = LogisticRegression(random_state=seed)
clf6 = DecisionTreeClassifier(random_state=seed)
clf7 = SVC(random_state=seed)

clf1.fit(trainDataGlobal, trainLabelsGlobal)
clf2.fit(trainDataGlobal, trainLabelsGlobal)
clf3.fit(trainDataGlobal, trainLabelsGlobal)
clf4.fit(trainDataGlobal, trainLabelsGlobal)
clf5.fit(trainDataGlobal, trainLabelsGlobal)
clf6.fit(trainDataGlobal, trainLabelsGlobal)
clf7.fit(trainDataGlobal, trainLabelsGlobal)

print(trainLabelsGlobal)

test_features = []
test_results = []

# for testing_name in test_labels:
#     # join the training data path and each species training folder
#     dir = os.path.join(test_path, testing_name)
#     # get the current training label
#     current_label = testing_name
#     # loop over the images in each sub-folder
#     for x in range(1, images_per_class + 1):
#         # get the image file name
#         index = random.randint(1, 80);
#         file = dir + "/" + "image (" + str(index) + ").jpg"
#         image = cv2.imread(file)
#         image = cv2.resize(image, fixed_size)
#         ####################################
#         fv_hu_moments = fd_hu_moments(image)
#         fv_haralick = fd_haralick(image)
#         fv_histogram = fd_histogram(image)
#
#         ###################################
#         test_results.append(current_label)
#         test_features.append(np.hstack([fv_histogram, fv_hu_moments, fv_haralick]))
for x in range(1, 51):
    current_label = train_labels[random.randint(0, 2)]
    index = random.randint(1, 80)
    file = test_path + "/" + current_label + " (" + str(index) + ").jpg"
    print(file)
    image = cv2.imread(file)
    image = cv2.resize(image, fixed_size)
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick = fd_haralick(image)
    fv_histogram = fd_histogram(image)

    ###################################
    test_results.append(current_label)
    test_features.append(np.hstack([fv_histogram, fv_hu_moments, fv_haralick]))


scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(test_features)

# predict label of test image
le = LabelEncoder()
y_result = le.fit_transform(test_results)
y_pred = clf1.predict(rescaled_features)
# print(y_result)
# print(y_pred)
print("RFC result: ", (y_pred == y_result).tolist().count(True)/len(y_result))

y_pred = clf2.predict(rescaled_features)
# print(y_pred)
print("KNN result: ", (y_pred == y_result).tolist().count(True)/len(y_result))

y_pred = clf3.predict(rescaled_features)
# print(y_pred)
print("GaussianNB result: ", (y_pred == y_result).tolist().count(True)/len(y_result))

y_pred = clf4.predict(rescaled_features)
# print(y_pred)
print("LR result: ", (y_pred == y_result).tolist().count(True)/len(y_result))

y_pred = clf5.predict(rescaled_features)
# print(y_pred)
print("LDA result: ", (y_pred == y_result).tolist().count(True)/len(y_result))

y_pred = clf6.predict(rescaled_features)
# print(y_pred)
print("DTC result: ", (y_pred == y_result).tolist().count(True)/len(y_result))

y_pred = clf7.predict(rescaled_features)
# print(y_pred)
print("SVM result: ", (y_pred == y_result).tolist().count(True)/len(y_result))


for x in range(1, 51):
    image_test_feature = []
    current_label = train_labels[random.randint(0, 2)]
    index = random.randint(1, 80)
    file = test_path + "/" + current_label + " (" + str(index) + ").jpg"
    print(file)
    image = cv2.imread(file)
    image = cv2.resize(image, fixed_size)
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick = fd_haralick(image)
    fv_histogram = fd_histogram(image)
    ###################################
    test_features.append(np.hstack([fv_histogram, fv_hu_moments, fv_haralick]))
    image_rescaled_features = scaler.fit_transform(test_features)
    y_pred = clf3.predict(image_rescaled_features)[-1]
    print(y_pred)
    cv2.putText(image, train_labels[y_pred], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    # display the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
# split the training and testing data
# (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features), np.array(global_labels), test_size=test_size, random_state=seed)
#
# print("[STATUS] splitted train and test data...")
# print("Train data  : {}".format(trainDataGlobal))
# print("Test data   : {}".format(testDataGlobal))
# print("Train labels: {}".format(trainLabelsGlobal))
# print("Test labels : {}".format(testLabelsGlobal))

# # 10-fold cross validation
# for name, model in models:
#     kfold = KFold(n_splits=10, random_state=seed)
#     cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)

# # boxplot algorithm comparison
# fig = pyplot.figure()
# fig.suptitle('Machine Learning algorithm comparison')
# ax = fig.add_subplot(111)
# pyplot.boxplot(results)
# ax.set_xticklabels(names)
# pyplot.show()

# -----------------------------------
# TESTING OUR MODEL
# -----------------------------------

# # to visualize results
# import matplotlib.pyplot as plt
#
# # create the model - Random Forests
# clf = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
#
# # fit the training data to the model
# clf.fit(trainDataGlobal, trainLabelsGlobal)
#
# # loop through the test images
# for file in glob.glob(test_path + "/*.jpg"):
#     # read the image
#     image = cv2.imread(file)
#
#     # resize the image
#     image = cv2.resize(image, fixed_size)
#
#     ####################################
#     # Global Feature extraction
#     ####################################
#     fv_hu_moments = fd_hu_moments(image)
#     fv_haralick = fd_haralick(image)
#
#     ###################################
#     # Concatenate global features
#     ###################################
#     global_feature = np.hstack([fv_haralick, fv_hu_moments])
#
#     # scale features in the range (0-1)
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     rescaled_feature = scaler.fit_transform(global_feature.reshape(1, -1))
#
#     # predict label of test image
#     prediction = clf.predict(rescaled_feature.reshape(1, -1))[0]
#
#     # show predicted label on image
#     cv2.putText(image, train_labels[prediction], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
#
#     # display the output image
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.show()