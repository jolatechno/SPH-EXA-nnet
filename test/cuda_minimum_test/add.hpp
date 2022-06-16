#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#define COMMA ,

#define CUDA_FUNCTION_DECORATOR __host__ __device__

#define CUDA_DEFINE(type, symbol, definition) \
		       type       symbol definition   \
	__device__ type dev_##symbol definition

#ifdef __CUDA_ARCH__ 
	#define CUDA_ACCESS(symbol) dev_##symbol
#else
	#define CUDA_ACCESS(symbol) symbol
#endif




namespace algorithm {
	/// equivalent to std::accumulate
	template<typename Float, class it>
	CUDA_FUNCTION_DECORATOR Float inline accumulate(const it begin, const it end, Float x) {
		for (it i = begin; i != end; ++i)
			x += *i;

		return x;
	}

	/// equivalent to std::swap
	template<typename Float>
	CUDA_FUNCTION_DECORATOR void inline swap(Float &x, Float &y) {
		Float buffer = x;
		x = y;
		y = buffer;
	}

	/// equivalent to std::fill
	template<typename Float, class it>
	CUDA_FUNCTION_DECORATOR void inline fill(it begin, it end, Float x) {
		for (it i = begin; i != end; ++i)
			*i = x;
	}

	/// equivalent to std::min
	template<typename Float>
	CUDA_FUNCTION_DECORATOR Float inline min(Float x, Float y) {
		if (x < y)
			return x;
		return y;
	}

	/// equivalent to std::min
	template<typename Float>
	CUDA_FUNCTION_DECORATOR Float inline max(Float x, Float y) {
		if (x > y)
			return x;
		return y;
	}
}





/// number of consecutive iteration per cuda thread
const int cuda_num_iteration_per_thread = 16;
/// number of thread per cuda thread block
const int cuda_num_thread_per_block = 128;




CUDA_DEFINE(const std::array<float COMMA 9>, offset, = {1 COMMA 3 COMMA 2 COMMA 5 COMMA 4 COMMA 7 COMMA 6 COMMA 9 COMMA 8};)


struct functor {
	template<typename Float>
	CUDA_FUNCTION_DECORATOR Float inline operator()(Float x, Float y, const Float *o, size_t i) {
		algorithm::swap(x, y);
		return std::max((Float)0.0, 
			x   +
			y*2 +
			algorithm::accumulate(o, o + i%10, (Float)0.0));
	}
};



// Kernel function to add the elements of two arrays
template<typename Float, class func_type>
__global__ void add(size_t n, Float *x, Float *y, func_type &f)
{
	size_t thread = blockIdx.x*blockDim.x + threadIdx.x;
	size_t begin  =       thread*cuda_num_iteration_per_thread;
	size_t end    = (thread + 1)*cuda_num_iteration_per_thread;
	if (end > n)
	    end = n;

	Float *local_offset = new Float[10];
	for (size_t i = 0; i < 10; i++)
		local_offset[i] = CUDA_ACCESS(offset)[i] + 1;

	for (size_t i = begin; i < end; i++)
    	y[i] = f(x[i], y[i], local_offset, i);

    delete[] local_offset;
}