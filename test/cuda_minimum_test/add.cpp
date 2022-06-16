#include <iostream>
#include <math.h>
#include <array>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>


#include "add.hpp"



int main(void)
{
	functor func;

	int N = 1<<20;




	/*float *x, *y;

	// Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged(&x, N*sizeof(float));
	cudaMallocManaged(&y, N*sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}*/





	// initialize a host_vector with the first five elements of D
    thrust::host_vector<float> x(N), y(N);
	thrust::device_vector<std::array<float, 2>> x_dev(N/2), y_dev(N/2);

	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	cudaMemcpy((void*)thrust::raw_pointer_cast(x_dev.data()), (void*)x.data(), N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)thrust::raw_pointer_cast(y_dev.data()), (void*)y.data(), N*sizeof(float), cudaMemcpyHostToDevice);








	// Run kernel on 1M elements on the GPU
	int num_threads     = (N           + cuda_num_iteration_per_thread - 1)/cuda_num_iteration_per_thread;
	int cuda_num_blocks = (num_threads + cuda_num_thread_per_block     - 1)/cuda_num_thread_per_block;

	add<<<cuda_num_blocks, cuda_num_thread_per_block>>>(N, 
		//x, y,
		(float*)thrust::raw_pointer_cast(x_dev.data()), (float*)thrust::raw_pointer_cast(y_dev.data()),
		func);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();





	cudaMemcpy((void*)y.data(), (void*)thrust::raw_pointer_cast(y_dev.data()), N*sizeof(float), cudaMemcpyDeviceToHost);





	float *local_offset = new float[10];
	for (size_t i = 0; i < 10; i++)
		local_offset[i] = CUDA_ACCESS(offset)[i] + 1;

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = fmax(maxError, fabs(y[i] - func(1.0f, 2.0f, local_offset, i)));
	std::cout << "Max error: " << maxError << std::endl;

	delete[] local_offset;

	for (int i = 0; i < 20; i++)
		std::cout << y[i] << ", ";
	std::cout << std::endl;





	// Free memory
	/*cudaFree(x);
	cudaFree(y);*/

	return 0;
}