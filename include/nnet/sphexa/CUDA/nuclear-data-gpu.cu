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
 * @brief Definition of CUDA functions for CUDA GPU data.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */


#include "nuclear-data-gpu.cuh"

namespace sphexa {
	namespace sphnnet {
		template<size_t n_species, typename Float, typename Int>
		void DeviceNuclearDataType<n_species, Float, Int>::resize(size_t size) {
	        double growthRate = 1;
	        auto   data_      = data();

	        for (size_t i = 0; i < data_.size(); ++i) {
	            if (this->isAllocated(i)) {
	            	// actually resize
	                std::visit([&](auto& arg) {
	        			using T = std::decay_t<decltype(*arg)>;

	        			if constexpr (std::is_convertible<typename T::value_type, int>::value) { // check for "normal" vectors (AKA not Y)
	                		reallocate(*arg, size, growthRate); 
	        			} else { // resize Y
	        				std::vector<Float*> Y_raw_ptr(n_species);

	        				for (int i = 0; i < n_species; ++i) {
	        					// reallocate Y
	                			reallocate(Y[i], size, growthRate);

	                			// store Y raw pointer to CPU
	                			Y_raw_ptr[i] = (Float*)thrust::raw_pointer_cast(Y[i].data());
	        				}

	        				// copy raw pointer to GPU
	        				Y_dev_ptr = Y_raw_ptr;
	        			}
	                }, data_[i]);
	            }
	        }
		}

		// used templates:
	    template class DeviceNuclearDataType<14, double, size_t>;
	    template class DeviceNuclearDataType<86, double, size_t>;
	    template class DeviceNuclearDataType<87, double, size_t>;

	    template class DeviceNuclearDataType<14, float, size_t>;
	    template class DeviceNuclearDataType<86, float, size_t>;
	    template class DeviceNuclearDataType<87, float, size_t>;
	}
}