import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from FeatureVectors import FeatureVectors
from QuerySearch import QuerySearch


def extractFeatureVectors(image_path):
    # Extracts feature vectors for input image

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (500, 500))
    featureVectors = FeatureVectors(image)
    vectors = featureVectors.getFeatureVector('host')

    imageName = image_path.split("/")[-1]
    return [imageName, vectors]

def chi2_distance(queryVector, vector, feaureMatrix, cosine_similarity):
    # Computes chi-square distance between two vectors
    eps = 1e-10
    for row in range(vector.shape[0]):
      # if col < queryVector.shape[0] and row < queryVector.shape[1]:
      temp = 0
      for i in range(len(queryVector)):

        temp += ((queryVector[i]- vector[row, i]) ** 2) / (queryVector[i] + vector[row, i] + eps)

      cosine_similarity[row, ] = temp * 0.5

def ImageSearch(queryImage):
    db_image_path = "Image_Database/"
    db_imageName_list = []
    db_features = []

    # Feature extraction database
    for img in os.listdir(db_image_path):
        imageName, vector = extractFeatureVectors(db_image_path+img)
        db_features.append(vector)
        db_imageName_list.append(img)
   
    db_features_Matrix = np.array(db_features)
    db_imageName_list = np.array(db_imageName_list)

    # Feature extraction testset
    ts_image_path = "Image_Testset/"
    ts_imageName_list = []
    ts_features = []

    # Feature extraction database
    for img in os.listdir(ts_image_path):
        imageName, vector = extractFeatureVectors(ts_image_path+img)
        ts_features.append(vector)
        ts_imageName_list.append(img)
            
    ts_features_Matrix = np.array(ts_features)
    ts_imageName_list = np.array(ts_imageName_list)

    true_val = 0
    total_val = 0
    incorrect_imageName = {}
    correct_imageName = {}

    true_val = 0
    total_val = 0

    incorrect_imageName = {}
    correct_imageName = {}

    for idx, queryVector in enumerate(ts_features_Matrix):
      # queryMatrix = np.array([queryVector,]*len(image_paths))
      cosine_similarity = np.zeros((db_features_Matrix.shape[0],))
      cosineMatrix = np.empty((db_features_Matrix.shape[0], db_features_Matrix.shape[1]))
      chi2_distance(queryVector, db_features_Matrix, cosineMatrix, cosine_similarity)

    
      # results = search.performSearch(use_device= True)
      queryName = ts_imageName_list[idx]
      index_result = np.argsort(cosine_similarity)[0]
      result = db_imageName_list[index_result]
      
      if queryName[:4] == result[:4]:
        correct_imageName[queryName] = result
        true_val+=1
      else:
        incorrect_imageName[queryName] = result

      total_val +=1

    return true_val, total_val,correct_imageName, incorrect_imageName
