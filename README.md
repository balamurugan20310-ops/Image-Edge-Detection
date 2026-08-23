# CUDA Sobel Edge Detection

## 📌 Project Overview

This project implements **Sobel Edge Detection using NVIDIA CUDA and OpenCV**.

The program reads a grayscale image, processes the image on the GPU using a CUDA kernel, detects edges using the Sobel operator, and saves the resulting edge-detected image.

The main purpose of this project is to demonstrate how **GPU parallel processing** can be used for image processing.

## 🚀 Technologies Used

* CUDA C/C++
* NVIDIA CUDA Toolkit
* OpenCV
* C++
* Google Colab
* NVIDIA GPU

## 📂 Project Structure

```text
CUDA-Sobel-Edge-Detection/
│
├── sobelEdgeDetectionFilter.cu
├── README.md
├── input/
│   └── gill.jpg
│
├── output/
│   └── output_sobel.jpeg
│
└── screenshots/
    └── result.png
```

## ⚙️ How It Works

The program follows these steps:

1. Read the input image using OpenCV.
2. Convert/read the image as grayscale.
3. Allocate memory on the CPU and GPU.
4. Copy the input image from CPU memory to GPU memory.
5. Launch the CUDA Sobel filter kernel.
6. Each CUDA thread processes one image pixel.
7. Calculate the horizontal gradient `Gx`.
8. Calculate the vertical gradient `Gy`.
9. Calculate the edge magnitude:

```text
Magnitude = sqrt(Gx² + Gy²)
```

10. Clamp the magnitude to the range `0–255`.
11. Copy the processed image back from GPU to CPU.
12. Save the result as `output_sobel.jpeg`.
13. Display the GPU execution time.

## 🧮 Sobel Operators

### Horizontal Gradient (Gx)

```text
[-1   0   1]
[-2   0   2]
[-1   0   1]
```

### Vertical Gradient (Gy)

```text
[-1  -2  -1]
[ 0   0   0]
[ 1   2   1]
```

The two gradients are combined to determine the strength of an edge.

## 🧵 CUDA Parallelization

The image is divided into CUDA threads using:

```cpp
dim3 blockSize(16, 16);
```

Each CUDA thread is responsible for processing one pixel.

The grid size is calculated according to the image dimensions:

```cpp
dim3 gridSize(
    (width + blockSize.x - 1) / blockSize.x,
    (height + blockSize.y - 1) / blockSize.y
);
```

This allows thousands of GPU threads to process different pixels simultaneously.

## 💻 Compilation

If CUDA and OpenCV are installed, compile using:

```bash
nvcc sobelEdgeDetectionFilter.cu -o sobelEdgeDetectionFilter `pkg-config --cflags --libs opencv4`
```

Run:

```bash
./sobelEdgeDetectionFilter
```

## 📊 Example Output

The program produces an output image containing the detected edges.

```text
Input Image
     ↓
Grayscale Image
     ↓
CUDA Sobel Kernel
     ↓
Gx + Gy Gradient Calculation
     ↓
Edge Magnitude
     ↓
Output Edge Image
```

## ⏱️ Performance Measurement

CUDA events are used to measure GPU kernel execution time.

Example:

```text
Total time taken: 0.XXX milliseconds
```

The execution time depends on the GPU, image resolution, CUDA configuration, and system environment.

## 🎯 Features

* GPU-based image processing
* CUDA parallel execution
* Sobel edge detection
* Grayscale image processing
* OpenCV image input/output
* CUDA execution-time measurement
* 16 × 16 CUDA thread blocks

## 📸 Sample Result

Add your input and output screenshots here.

```text
Input:  gill.jpg
Output: output_sobel.jpeg
```

## 🔮 Future Improvements

* Process multiple images in a batch.
* Compare CPU and GPU execution times.
* Implement Gaussian blur before Sobel filtering.
* Add RGB image support.
* Optimize the kernel using CUDA shared memory.
* Calculate speedup between CPU and GPU implementations.

## 👨‍💻 Author

**Balamurugan S**

B.Tech Artificial Intelligence and Machine Learning

Saveetha Engineering College

## Program

```c
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter

%load_ext nvcc4jupyter

from pathlib import Path

file_path = Path('/absolute/path/to/images.jpeg')
if file_path.exists():
    print("File exists!")
else:
    print("File does not exist!")

import os
print("Current Working Directory:", os.getcwd())

from google.colab import files
uploaded = files.upload()

from pathlib import Path

# Assuming the file is in the same directory as the notebook
file_path = Path('gill.jpg')
if file_path.exists():
    print("File exists!")
else:
    print("File does not exist!")

pwd

ls /content/gill.jpg

#ls -l /content/images.jpeg
import cv2
image = cv2.imread('/content/gill.jpg')
if image is None:
    print("Error: Image not found or unable to read the image.")
else:
    print("Image read successfully.")

%%writefile sobelEdgeDetectionFilter.cu
#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;

// CUDA Kernel for Sobel Edge Detection
__global__ void sobelFilter(unsigned char *srcImage, unsigned char *dstImage, unsigned int width, unsigned int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check for thread layout
    if (x >= width || y >= height) return;

    // Set boundary pixels to black (0) to avoid out-of-bounds array access
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        dstImage[y * width + x] = 0;
        return;
    }

    // Horizontal gradient operator (Gx)
    // [-1  0  1]
    // [-2  0  2]
    // [-1  0  1]
    int Gx = -1 * srcImage[(y - 1) * width + (x - 1)] + 1 * srcImage[(y - 1) * width + (x + 1)]
             - 2 * srcImage[y       * width + (x - 1)] + 2 * srcImage[y       * width + (x + 1)]
             - 1 * srcImage[(y + 1) * width + (x - 1)] + 1 * srcImage[(y + 1) * width + (x + 1)];

    // Vertical gradient operator (Gy)
    // [-1 -2 -1]
    // [ 0  0  0]
    // [ 1  2  1]
    int Gy = -1 * srcImage[(y - 1) * width + (x - 1)] - 2 * srcImage[(y - 1) * width + x] - 1 * srcImage[(y - 1) * width + (x + 1)]
             + 1 * srcImage[(y + 1) * width + (x - 1)] + 2 * srcImage[(y + 1) * width + x] + 1 * srcImage[(y + 1) * width + (x + 1)];

    // Compute edge magnitude: sqrt(Gx^2 + Gy^2)
    int magnitude = sqrtf((float)(Gx * Gx + Gy * Gy));

    // Clamp value to [0, 255] range
    if (magnitude > 255) magnitude = 255;

    dstImage[y * width + x] = (unsigned char)magnitude;
}

void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Read input image as grayscale
    Mat image = imread("/content/gill.jpg", IMREAD_GRAYSCALE);

    if (image.empty()) {
        printf("Error: Image not found.\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    // Allocate host memory for output image
    unsigned char *h_outputImage = (unsigned char *)malloc(imageSize);
    if (h_outputImage == nullptr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // Allocate device memory
    unsigned char *d_inputImage, *d_outputImage;
    checkCudaErrors(cudaMalloc(&d_inputImage, imageSize));
    checkCudaErrors(cudaMalloc(&d_outputImage, imageSize));
    checkCudaErrors(cudaMemcpy(d_inputImage, image.data, imageSize, cudaMemcpyHostToDevice));

    // Define CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    cudaEventRecord(start);
    sobelFilter<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);
    cudaEventRecord(stop);

    // Synchronize events
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost));

    // Write output image
    Mat outputImage(height, width, CV_8UC1, h_outputImage);
    imwrite("output_sobel.jpeg", outputImage);

    // Free memory
    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print elapsed time
    printf("Total time taken: %f milliseconds\n", milliseconds);

    return 0;
}

!./sobelEdgeDetectionFilter

import cv2
from matplotlib import pyplot as plt

# Read and display the output image
output_image_path = '/content/output_sobel.jpeg'
output_image = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)  # Use IMREAD_GRAYSCALE if it's a single-channel image

# Display the image
plt.imshow(output_image, cmap='gray')
plt.title('Edge Detection Output')
plt.axis('off')  # Hide the axes
plt.show()

```

## Output
<img width="1404" height="484" alt="image" src="https://github.com/user-attachments/assets/d7bfb420-32b2-48a0-9247-5de3da207c51" />



## Result 
  Sobel edge detection was successfully implemented using CUDA GPU parallel processing.
The input grayscale image was processed using Gx and Gy gradients to detect edges.
The final edge-detected image was saved as output_sobel.jpeg.
