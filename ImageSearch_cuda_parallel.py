import os
import sys
import threading
import concurrent.futures
import time
import cv2
import matplotlib.pyplot as plt
from FeatureVectors import FeatureVectors
from QuerySearch import QuerySearch
from numba import cuda, prange
import numpy as np
import math
from numba import cuda, types as numba_types
import numba


def extractFeatureVectors(image_path):
    # Extracts feature vectors for input image

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (500, 500))
    featureVectors = FeatureVectors(image)
    vectors = featureVectors.getFeatureVector('cuda_device')

    imageName = image_path.split("/")[-1]
    return [imageName, vectors]

def getImg(img):
    # image_db_path = "Image_Database/"
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(img)
    return image

@cuda.jit
def chi2_distance_kernel_cuda( d_queryMatrix, d_db_features_Matrix, d_cosineMatrix, d_cosine_similarity):
  eps = 1e-10
  col, row = cuda.grid(2)

  if col >= d_db_features_Matrix.shape[0] and row >= d_db_features_Matrix.shape[1]:
    return

  temp = 0
  for i in range(len(d_queryMatrix)):
    temp += ((d_queryMatrix[i]- d_db_features_Matrix[row, i]) ** 2) / \
            (d_queryMatrix[i] + d_db_features_Matrix[row, i] + eps)

  d_cosine_similarity[row,] = temp * 0.5


def CUDAParallelImageSearch(queryImage):
    # Performs Image Search using Query image
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

    # queryImage_path = image_db_path+queryImage

    # imageName, queryVector = extractFeatureVectors(queryImage_path)

    # Feature extraction testset
    ts_image_path = queryImage
    ts_imageName_list = []
    ts_features = []

    # Feature extraction database
    for img in os.listdir(ts_image_path):
        imageName, vector = extractFeatureVectors(ts_image_path+img)
        ts_features.append(vector)
        ts_imageName_list.append(img)

            
    ts_features_Matrix = np.array(ts_features)
    ts_imageName_list = np.array(ts_imageName_list)

    # -----

    block_size = (32, 32)
    grid_size = (math.ceil(db_features_Matrix.shape[1]/ block_size[0]), 
                math.ceil(db_features_Matrix.shape[0] / block_size[1]))
    
    true_val = 0
    total_val = 0

    incorrect_imageName = {}
    correct_imageName = {}
    d_db_features_Matrix = cuda.to_device(db_features_Matrix)
    d_ts_features_Matrix = cuda.to_device(ts_features_Matrix)
    for idx, d_queryVector in enumerate(d_ts_features_Matrix):
      # queryMatrix = np.array([queryVector,]*db_features_Matrix.shape[0])
      # d_queryMatrix = cuda.to_device(queryMatrix)
      # d_queryVector = cuda.to_device(queryVector)
      cosine_similarity = cuda.device_array((d_db_features_Matrix.shape[0],), dtype=np.float64)
      d_cosineMatrix = cuda.device_array((db_features_Matrix.shape[0], db_features_Matrix.shape[1]))

      chi2_distance_kernel_cuda[grid_size, block_size](d_queryVector, d_db_features_Matrix, d_cosineMatrix, cosine_similarity)
      # cuda.synchronize()
      # cosine_similarity = d_cosine_similarity.copy_to_host()
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

    return true_val, total_val, correct_imageName, incorrect_imageName