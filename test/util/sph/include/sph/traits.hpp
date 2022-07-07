#pragma once

#include <type_traits>

namespace cstone {
    struct CpuTag
    {
    };
    struct GpuTag
    {
    };
}

namespace sphexa {
    template<class AccType>
    struct HaveGpu : 
        public std::integral_constant<int, std::is_same_v<AccType, cstone::GpuTag>>
    {
    };
}