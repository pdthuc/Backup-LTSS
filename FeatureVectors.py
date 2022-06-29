import cv2
import numpy as np
from Image_Filters.NoiseReduction import NoiseReduction
from Image_Filters.ConvolutionalFilters import ConvolutionFilter
from numba import njit 

@njit
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result

@njit
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)

@njit
def np_std(array, axis):
        return np_apply_along_axis(np.std, axis, array)

class FeatureVectors():
    """Extracts various feature vectors from image """

    def __init__(self, image):
        self.filter = NoiseReduction(image)
        self.image = self.filter.applyGaussianBlur()

    def __getMeanIntensity(self, use_device):
        # Returns the mean intensity of image
        if use_device=='host':
            meanIntensity = []
            for channel in range(3):
                channel_mean = np.average(self.image[:, :, channel])
                meanIntensity.append(round(channel_mean, 5))
        else:
            image=self.image
            image_flatten = image.reshape(image.shape[0]*image.shape[1], image.shape[2])
            meanIntensity = list(np_mean(image_flatten, 0))

        return meanIntensity

    def __getStdIntensity(self, use_device):
        # Returns the standard deviation of intensity of image
        stdIntensity = []

        if use_device=='host':
            stdIntensity = []
            for channel in range(3):
                channel_std = np.std(self.image[:, :, channel])
                stdIntensity.append(round(channel_std, 5))
        else:
            image=self.image
            image_flatten = image.reshape(image.shape[0]*image.shape[1], image.shape[2])
            stdIntensity = list(np_std(image_flatten, 0))

        return stdIntensity

    def __getRGBHistogramVector(self):
        # Returns the 3-D(RGB) Histogram vector of image

        histogram_3d = cv2.calcHist([self.image], [0, 1, 2], None,
                                    [12, 12, 12], [0, 256, 0, 256, 0, 256])
        histogram_3d = histogram_3d.ravel()
        RGBHistogram = list(histogram_3d)

        return RGBHistogram

    def __getHuMoments(self, use_device):
        # Returns Hu-Moments vector of image
        laplacian_filter = ConvolutionFilter(self.image)
        laplacian_filtered = laplacian_filter.applyLaplacian(use_device)

        canny_huMoments = cv2.HuMoments(cv2.moments(laplacian_filtered)).flatten()
        huVector = list(canny_huMoments.ravel())
        return huVector

    def getFeatureVector(self, use_device):
        """ Return a python list of complete feature vectors
        Extracts Statistics, 3-D Histogram, HuMoments from image and appends into single list
        """
        featureVectors = []
        meanIntensity = self.__getMeanIntensity(use_device)
        stdIntensity = self.__getStdIntensity(use_device)
        rgbHistogram = self.__getRGBHistogramVector()
        huVectors = self.__getHuMoments(use_device)

            
        colorVectors = meanIntensity+stdIntensity+rgbHistogram
        featureVectors = colorVectors+huVectors

        return np.array(featureVectors)