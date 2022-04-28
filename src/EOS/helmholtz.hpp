#pragma once

#include <vector>
#include <tuple>
#include <fstream>

namespace nnet::eos {
	namespace helmotz_constants {
		// read helmotz constants table
		auto const [table1, table2 /* TODO */] = []() {
			// table size
	   		const int imax = 541, jmax = 201;
	   		
			/* TODO ... */
	   		static double d[imax], t[jmax];

	   		// read file
			std::ifstream hem_table; 
	   		hem_table.open("hem_table.dat");

	   		// standard table limits
			const double tlo   = 3.0;
			const double  thi   = 13.0;
			const double  tstp  = (thi - tlo)/(double)(jmax-1);
			const double  tstpi = 1.0/tstp;
			const double  dlo   = -12.0;
			const double  dhi   = 15.0;
			const double  dstp  = (dhi - dlo)/(double)(imax-1);
			const double  dstpi = 1.0/dstp;


			// read the helmholtz free energy and its derivatives
			for (int j = 0; j < jmax; ++j) {
				/* TODO ... */
			}

			// read the pressure derivative with density table
			for (int j = 0; j < jmax; ++j)
				for (int i = 0; i < imax; ++i) {
					/* TODO ... */
				}

			// read the electron chemical potential table
			for (int j = 0; j < jmax; ++j)
				for (int i = 0; i < imax; ++i) {
					/* TODO ... */
				}

			// read the number density table
			for (int j = 0; j < jmax; ++j)
				for (int i = 0; i < imax; ++i) {
					/* TODO ... */
				}

			// construct the temperature and density deltas and their inverses
			for (int j = 0; j < jmax - 1; ++j) {
				const double dth  = t[j+1] - t[j];
				const double dt2  = dth*dth;
				const double dti  = 1.0/dth;
				const double dt2i = 1.0/dt2;
				const double dt3i = dt2i*dti;

				/* TODO ... */
			}

			// construct the temperature and density deltas and their inverses
			for (int i = 0; i < imax - 1; ++i) {
				const double dd   = d[i+1] - d[i];
				const double dd2  = dd*dd;
				const double ddi  = 1.0/dd;
				const double dd2i = 1.0/dd2;
				const double dd3i = dd2i*ddi;

				/* TODO ... */
			}
	   		
	   		return std::pair<double*, double*>{d, t};
		}();
	}



	/// helmotz eos
	/**
	 * ...TODO
	 */
	template<typename Float>
	class helmotz {
	private:
		double rho = 1e9;

	public:
		helmotz(Float initial_rho) : rho(initial_rho) {
			/* TODO */
		}

		std::tuple<Float, Float, Float>operator()(const std::vector<Float> &Y, const Float T) {
			/* TODO */
			return std::tuple<Float, Float, Float>{/*cv*/2e7, rho, /*value_1*/0};
		}
	};
}