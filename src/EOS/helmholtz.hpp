#pragma once

#include <vector>

namespace nnet::eos {
	template<typename Float>
	std::tuple<Float, Float, Float>helmotz(const std::vector<Float> &Y, const Float T) {
		/* TODO */
		return std::tuple<Float, Float, Float>{/*cv*/2e7, /*rho*/1e9, /*value_1*/0};
	};
}