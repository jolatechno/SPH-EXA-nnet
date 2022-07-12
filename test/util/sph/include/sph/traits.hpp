#pragma once

#include <type_traits>
#include <array>

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

    //! @brief stub for use in CPU code
    template<class T, class KeyType>
    struct DeviceDataFacade {
        void resize(size_t) {}

        template<class... Ts>
        void setConserved(Ts...)
        {
        }

        template<class... Ts>
        void setDependent(Ts...)
        {
        }

        template<class... Ts>
        void release(Ts...)
        {
        }

        template<class... Ts>
        void acquire(Ts...)
        {
        }

        inline static constexpr std::array fieldNames{0};
    };

    namespace detail {
        //! @brief The type member of this trait evaluates to CpuCaseType if Accelerator == CpuTag and GpuCaseType otherwise
        template<class Accelerator, template<class...> class CpuCaseType, template<class...> class GpuCaseType, class = void>
        struct AccelSwitchType
        {
        };

        template<class Accelerator, template<class...> class CpuCaseType, template<class...> class GpuCaseType>
        struct AccelSwitchType<Accelerator, CpuCaseType, GpuCaseType, std::enable_if_t<!HaveGpu<Accelerator>{}>>
        {
            template<class... Args>
            using type = CpuCaseType<Args...>;
        };

        template<class Accelerator, template<class...> class CpuCaseType, template<class...> class GpuCaseType>
        struct AccelSwitchType<Accelerator, CpuCaseType, GpuCaseType, std::enable_if_t<HaveGpu<Accelerator>{}>>
        {
            template<class... Args>
            using type = GpuCaseType<Args...>;
        };
    } // namespace detail
}