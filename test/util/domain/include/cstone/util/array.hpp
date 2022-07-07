#include <array>

namespace util {
	template<class T, long unsigned int n>
	using array = std::array<T, n>;
}