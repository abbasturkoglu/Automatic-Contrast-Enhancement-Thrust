// Do not alter the preprocessor directives
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <driver_types.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cassert>
#include <cstdlib>
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/pair.h"
#include "thrust/extrema.h"


#define NUM_CHANNELS 1

#define MinVal(x, y) (((x) < (y)) ? (x) : (y))
#define MaxVal(x, y) (((x) > (y)) ? (x) : (y))

//This function multiply every pixel value with scale constant
struct multiplyFunction
{
	float a;
 

	multiplyFunction(float s_constant) {
		a = s_constant; 
	}

	__host__ __device__
		uint8_t operator()(const uint8_t& x) const
	{
		return a*x ;
	}
};

//This function is another option to contrast image with one epoch. Subtracts the minimum value and multiply with scale constant together
struct scaleFunction
{
	unsigned int a;
 unsigned int b;

	scaleFunction(unsigned int s_constant,unsigned int min) {
		a = s_constant; 
    b= min;
	}

	__host__ __device__
		uint8_t operator()(const uint8_t& x) const
	{
		return (x-b)*a ;
	}
};

int main() {

	int width; //image width
	int height; //image height
	int bpp;  //bytes per pixel if the image was RGB (not used)


	// Load a grayscale bmp image to an unsigned integer array with its height and weight.
	//  (uint8_t is an alias for "unsigned char")
  uint8_t* image =  stbi_load("./samples/640x426.bmp", &width, &height, &bpp, NUM_CHANNELS);
  size_t image_size = width * height * sizeof(uint8_t);




	// Print for sanity check
	printf("Bytes per pixel: %d \n", bpp / 3); //Image is grayscale, so bpp / 3;
	printf("Height: %d \n", height);
	printf("Width: %d \n", width);


	//Start Counter
	cudaEvent_t start, stop;
	float elapsed_time_ms;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);


  //Create device vector image_d and initialize with value of image
  thrust::device_vector<uint8_t> image_d(image, image + (width * height));

  // Find minimum and maximum values
  int min_t = thrust::reduce(image_d.begin(), image_d.end(),255, thrust::minimum<int>());
  int max_t = thrust::reduce(image_d.begin(), image_d.end(), 0, thrust::maximum<int>());
  
  
  float scale_constant = 255.0f / (max_t - min_t);

  // I designed 3 different thrust kernels. They do the same job with different methods
  // Their performances are very similar so it does not matter which one you use
  
  //option 1 for subtract and scale
  thrust::for_each(image_d.begin(), image_d.end(), thrust::placeholders::_1 -= min_t);
  thrust::transform(image_d.begin(), image_d.end(), image_d.begin(), multiplyFunction(scale_constant));

  //option 2 for subtract and scale
  //thrust::for_each(image_d.begin(), image_d.end(), thrust::placeholders::_1 -= min_t);
  //thrust::for_each(image_d.begin(), image_d.end(), thrust::placeholders::_1 *= scale_constant);


  //option 3 for subtract and scale
  //thrust::transform(image_d.begin(), image_d.end(), image_d.begin(), scaleFunction(scale_constant,min_t));

  //Copy enhanced image to host
  thrust::host_vector<uint8_t> image_h(image_d);
	uint8_t* image_e = image_h.data();



	//Stop timing
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	printf("\nTime to calculate results(GPU Time): %f ms.\n\n", elapsed_time_ms);


	// Write image array into a bmp file
	stbi_write_bmp("./samples/out_img.bmp", width, height, 1, image_e);
	printf("\nEnchanced image successfully saved.\n\n");

  //print minimum and maximum value
	printf("Minimum Pixel Value: %d\n", min_t);
	printf("Maximum Pixel Value: %d\n", max_t);



	return 0;
}