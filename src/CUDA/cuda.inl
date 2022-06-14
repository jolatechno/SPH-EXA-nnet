#define COMMA ,

#ifdef USE_CUDA
	#define CUDA_FUNCTION_DECORATOR __host__ __device__

	#define CUDA_DEFINE(type, symbol, definition) \
		           type     symbol definition      \
		__device__ type dev_##symbol definition
#else
	#define CUDA_FUNCTION_DECORATOR

	#define CUDA_DEFINE(type, symbol, definition) \
		type symbol definition
#endif

#ifdef __CUDA_ARCH__ 
	#define CUDA_ACCESS(symbol) dev_##symbol
	#define CUDA_ACCESS_MATRIX(symbol, i, j) dev_##symbol[i][j]
#else
	#define CUDA_ACCESS(symbol) symbol
	#define CUDA_ACCESS_MATRIX(symbol, i, j) symbol(i, j)
#endif