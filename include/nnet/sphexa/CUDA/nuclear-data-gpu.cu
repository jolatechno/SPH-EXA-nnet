#ifdef IMPORT_DOT_CU
#pragma once
#endif

#include "nuclear-data-gpu.cuh"

namespace sphexa {
	namespace sphnnet {
		template<class AccType, size_t n_species, typename Float>
		void DeviceNuclearDataType<AccType, n_species, Float>::resize(size_t size) {
	        double growthRate = 1;
	        auto   data_      = data();

	        for (size_t i = 0; i < data_.size(); ++i) {
	            if (this->isAllocated(i)) {
	            	// actually resize
	                std::visit([&](auto& arg) { 
	                	size_t previous_size = arg->size();
	                	reallocate(*arg, size, growthRate); 
	                }, data_[i]);
	            }
	        }
	    }
	}
}