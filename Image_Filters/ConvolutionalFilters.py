import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from numba import cuda, float32, uint8, int64, float64, njit, prange
import numba
import math

mask_rows, mask_cols = 3, 3
mask_size = mask_rows * mask_cols
delta_rows = mask_rows // 2
delta_cols = mask_cols // 2

# We use blocks of 32x32 pixels:
blockdim = (32, 32)

# We use an image of size:
image_rows, image_cols = 500, 500

# We compute grid dimensions big enough to cover the whole image:
griddim = (image_rows // blockdim[0] + 1,
          image_cols // blockdim[1] + 1)

# We want to keep in shared memory a part of the image of shape:
shared_image_rows = blockdim[0] + mask_rows - 1
shared_image_cols = blockdim[1] + mask_cols - 1
shared_image_size = shared_image_rows * shared_image_cols

class ConvolutionFilter():
    """Converts input image to grayscale and applies various convolution filters"""

    def __init__(self, image):
        self.image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        self.sharpen = np.array(([0, -1, 0],
                                 [-1,  5, -1],
                                 [0, -1, 0]))

        self.sobelX = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])

        self.sobelY = np.array(([-1, -2, -1],
                                [0,  0,  0],
                                [1,  2,  1]))

        self.laplacian = np.array(([0,  1, 0],
                                   [1, -4, 1],
                                   [0,  1, 0]))

    def __convolution(self, image_roi, kernel):
        # This function convolves the input kernel on the input image region of interest 

        kernel_dimension = len(kernel)
        pixel_sum = 0

        for i in range(kernel_dimension):
            for j in range(kernel_dimension):
                pixel_kernel_value = image_roi[i, j]*kernel[i, j]
                pixel_sum = pixel_sum+pixel_kernel_value

        if pixel_sum < 0:
            return 0
        else:
            return pixel_sum % 255

    def __applyFilter(self, kernel):
        """ Returns convolved image 
        Applies the input convolution filter onto the image 
        """

        image = self.image
        filtered_image = np.zeros(image.shape)
        image_rows, image_cols = image.shape
        delta_rows = kernel.shape[0] // 2 
        delta_cols = kernel.shape[1] // 2
        for i in range(image_rows):
          for j in range(image_cols):
            s = 0
            for k in range(kernel.shape[0]):
                for l in range(kernel.shape[1]):
                    i_k = i - k + delta_rows
                    j_l = j - l + delta_cols
                    if (i_k >= 0) and (i_k < image_rows) and (j_l >= 0) and (j_l < image_cols):  
                        s += kernel[k, l] * image[i_k, j_l]
            if s > 0:
              filtered_image[i, j] = s % 255
            else:
              filtered_image[i, j] = 0

        return filtered_image

    @cuda.jit
    def kernelApplyFilter(result, mask, image):
       
      i, j = cuda.grid(2) 
      
      image_rows, image_cols = image.shape
      if (i >= image_rows) or (j >= image_cols): 
          return
      
      delta_rows = mask.shape[0] // 2 
      delta_cols = mask.shape[1] // 2
      
      s = 0
      for k in range(mask.shape[0]):
          for l in range(mask.shape[1]):
              i_k = i - k + delta_rows
              j_l = j - l + delta_cols
              if (i_k >= 0) and (i_k < image_rows) and (j_l >= 0) and (j_l < image_cols):  
                  s += mask[k, l] * image[i_k, j_l]
      if s > 0:
        result[i, j] = s % 255
      else:
        result[i, j] = 0

    @cuda.jit
    def smem_convolve(result, mask, image):
      i, j = cuda.grid(2)

      shared_image = cuda.shared.array(shared_image_size, numba.types.float32)
      shared_mask = cuda.shared.array(mask_size, numba.types.float32)

      if (cuda.threadIdx.x < mask_rows) and (cuda.threadIdx.y < mask_cols):
          shared_mask[cuda.threadIdx.x + cuda.threadIdx.y * mask_rows] = mask[cuda.threadIdx.x, cuda.threadIdx.y]

      row_corner = cuda.blockDim.x * cuda.blockIdx.x - delta_rows
      col_corner = cuda.blockDim.y * cuda.blockIdx.y - delta_cols
      even_idx_x = 2 * cuda.threadIdx.x
      even_idx_y = 2 * cuda.threadIdx.y
      odd_idx_x = even_idx_x + 1
      odd_idx_y = even_idx_y + 1
      for idx_x in (even_idx_x, odd_idx_x):
          if idx_x < shared_image_rows:
              for idx_y in (even_idx_y, odd_idx_y):
                  if idx_y < shared_image_cols:
                      point = (row_corner + idx_x, col_corner + idx_y)
                      if (point[0] >= 0) and (point[1] >= 0) and (point[0] < image_rows) and (point[1] < image_cols):
                          shared_image[idx_x + idx_y * shared_image_rows] = image[point]
                      else:
                          shared_image[idx_x + idx_y * shared_image_rows] = numba.types.float32(0)
      cuda.syncthreads()

      s = 0
      for k in prange(mask_rows):
          for l in prange(mask_cols):
              i_k = cuda.threadIdx.x - k + mask_rows - 1
              j_l = cuda.threadIdx.y - l + mask_cols - 1
              s += shared_mask[k + l * mask_rows] * shared_image[i_k + j_l * shared_image_rows]

      if (i < image_rows) and (j < image_cols):
        if s > 0:
          result[i, j] = s % 255
        else:
          result[i, j] = 0

    def applySharpen(self):
        """Returns image convolved with Sharpening filter"""
        kernel = self.sharpen
        filtered_image = self.__applyFilter(kernel)
        return filtered_image

    def applySobelX(self):
        """Returns image convolved with SobelX filter"""
        kernel = self.sobelX
        filtered_image = self.__applyFilter(kernel)
        return filtered_image

    def applySobelY(self):
        """Returns image convolved with SobelY filter"""
        kernel = self.sobelY
        filtered_image = self.__applyFilter(kernel)
        return filtered_image

    def applyLaplacian(self, use_device = 'host'):
        """Returns image convolved with Laplacian filter"""
        image = self.image
        kernel = self.laplacian
        filtered_img = np.zeros((image.shape[0],image.shape[1]))
        block_size = (32, 32)
        grid_size = (math.ceil(image.shape[1] / block_size[0]), 
            math.ceil(image.shape[0] / block_size[1]))

        if use_device == 'host':
          filtered_img = self.__applyFilter(kernel)

        elif use_device == 'cuda_host':
          
          d_filtered_img = cuda.device_array((image.shape[0],image.shape[1]), dtype=np.float32)
          d_kernel = cuda.to_device(kernel)
          d_image = cuda.to_device(image)
          time.sleep(0.1)
          self.kernelApplyFilter[grid_size, block_size](d_filtered_img, d_kernel, d_image)
          cuda.synchronize()
          filtered_img = d_filtered_img.copy_to_host()

        elif use_device == 'cuda_device':
          d_filtered_img = cuda.device_array((image.shape[0],image.shape[1]), dtype=np.float32)
          d_kernel = cuda.to_device(kernel)
          d_image = cuda.to_device(image)
          
          self.smem_convolve[grid_size, block_size](d_filtered_img, d_kernel, d_image)
          cuda.synchronize()
          filtered_img = d_filtered_img.copy_to_host()

        return filtered_img

    def applyCannyEdge(self):
        """Returns image convolved with CannyEdge filter"""

        filtered_image = cv2.Canny(self.image, 50, 240)
        return filtered_image
