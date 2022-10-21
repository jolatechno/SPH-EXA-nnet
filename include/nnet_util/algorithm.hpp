/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Utility functions.
 * Mostly stolen from jolatechno/QuIDS (https://github.com/jolatechno/QuIDS).
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */


#pragma once

#include <vector>
#include <parallel/algorithm>
#include <parallel/numeric>

#include <omp.h>

#include "CUDA/cuda.inl"
#if COMPILE_DEVICE
	#include <cuda_runtime.h>
#endif


namespace algorithm {
	/*! @brief equivalent to std::accumulate but from scratch for potential device use */
	template<typename Float, class it>
	HOST_DEVICE_FUN Float inline accumulate(const it begin, const it end, Float x) {
		for (it i = begin; i != end; ++i)
			x += *i;

		return x;
	}

	/*! @brief equivalent to std::swap but from scratch for potential device use */
	template<typename Float>
	HOST_DEVICE_FUN void inline swap(Float &x, Float &y) {
		Float buffer = x;
		x = y;
		y = buffer;
	}

	/*! @brief equivalent to std::fill but from scratch for potential device use */
	template<typename Float, class it>
	HOST_DEVICE_FUN void inline fill(it begin, it end, Float x) {
		for (it i = begin; i != end; ++i)
			*i = x;
	}
}


namespace util {
	/*! @brief scheduling batch size according to an empirical rule */
	int inline dynamic_batch_size(size_t N, int P) {
		static const float phi = 1.61803398875; // golden ratio

		int f = (int)std::floor(std::log2((float)N/(float)P)/phi);
		int pow2_f = 1 << f;
		int batch_size = (int)std::floor((float)N/((float)pow2_f*2.0*(float)P));

		return std::max(1, batch_size);
	}
}
