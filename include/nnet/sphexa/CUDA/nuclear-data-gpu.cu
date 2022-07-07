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

	    template class DeviceNuclearDataType<cstone::GpuTag, 14, double>;
	    template class DeviceNuclearDataType<cstone::GpuTag, 86, double>;
	    template class DeviceNuclearDataType<cstone::GpuTag, 87, double>;

	    template class DeviceNuclearDataType<cstone::CpuTag, 14, double>;
	    template class DeviceNuclearDataType<cstone::CpuTag, 86, double>;
	    template class DeviceNuclearDataType<cstone::CpuTag, 87, double>;

	    template class DeviceNuclearDataType<cstone::GpuTag, 14, float>;
	    template class DeviceNuclearDataType<cstone::GpuTag, 86, float>;
	    template class DeviceNuclearDataType<cstone::GpuTag, 87, float>;

	    template class DeviceNuclearDataType<cstone::CpuTag, 14, float>;
	    template class DeviceNuclearDataType<cstone::CpuTag, 86, float>;
	    template class DeviceNuclearDataType<cstone::CpuTag, 87, float>;
	}
}