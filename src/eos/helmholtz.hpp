#pragma once

#include <iostream>

#include <sstream>
#include <string>
#include <fstream>

#include <vector>
#include <tuple>
#include <math.h>

#ifndef IMAX
	#define IMAX 541
#endif
#ifndef JMAX
	#define JMAX 201
#endif


namespace nnet::eos {
	namespace helmotz_constants {
		// table size
		const int imax = IMAX, jmax = JMAX;

		// tables
		double d[imax], dd_sav[imax], dd2_sav[imax], ddi_sav[imax], dd2i_sav[imax], dd3i_sav[imax],
			t[jmax], dt_sav[jmax], dt2_sav[jmax], dti_sav[jmax], dt2i_sav[jmax], dt3i_sav[jmax],
	   		f[imax][jmax],
	   		fd[imax][jmax], ft[imax][jmax],
	   		fdd[imax][jmax], ftt[imax][jmax], fdt[imax][jmax],
	   		fddt[imax][jmax], fdtt[imax][jmax], fddtt[imax][jmax],
	   		dpdf[imax][jmax], dpdfd[imax][jmax], dpdft[imax][jmax], dpdfdt[imax][jmax],
	   		ef[imax][jmax], efd[imax][jmax], eft[imax][jmax], efdt[imax][jmax],
	   		xf[imax][jmax], xfd[imax][jmax], xft[imax][jmax], xfdt[imax][jmax];

		// read helmotz constants table
		void read_table(const char *file_path){
	   		// read file
			std::ifstream helm_table; 
	   		helm_table.open(file_path);
	   		if (!helm_table) {
        		std::cerr << "Helm. table not found !\n";
        		throw;
	   		}

	   		// standard table limits
			const double tlo   = 3.0;
			const double thi   = 13.0;
			const double tstp  = (thi - tlo)/(double)(jmax-1);
			const double tstpi = 1.0/tstp;
			const double dlo   = -12.0;
			const double dhi   = 15.0;
			const double dstp  = (dhi - dlo)/(double)(imax-1);
			const double dstpi = 1.0/dstp;


			// read the helmholtz free energy and its derivatives
			for (int i = 0; i < imax; ++i) {
				const double dsav = dlo + (i - 1)*dstp;
				d[i] = std::pow(10., dsav);
			}
			for (int j = 0; j < jmax; ++j) {
				const double tsav = tlo + (j - 1)*tstp;
				t[j] = std::pow(10., tsav);

				for (int i = 0; i < imax; ++i) {
					helm_table >> f[i][j] >> fd[i][j] >> ft[i][j] >>
			   			fdd[i][j] >> ftt[i][j] >> fdt[i][j] >>
			   			fddt[i][j] >> fdtt[i][j] >> fddtt[i][j];
				}
			}

			// read the pressure derivative with density table
			for (int j = 0; j < jmax; ++j)
				for (int i = 0; i < imax; ++i) {
					helm_table >> dpdf[i][j] >> dpdfd[i][j] >> dpdft[i][j] >> dpdfdt[i][j];
				}

			// read the electron chemical potential table
			for (int j = 0; j < jmax; ++j)
				for (int i = 0; i < imax; ++i) {
					helm_table >> ef[i][j] >> efd[i][j] >> eft[i][j] >> efdt[i][j];
				}

			// read the number density table
			for (int j = 0; j < jmax; ++j)
				for (int i = 0; i < imax; ++i) {
					helm_table >> xf[i][j] >> xfd[i][j] >> xft[i][j] >> xfdt[i][j];
				}

			// construct the temperature and density deltas and their inverses
			for (int j = 0; j < jmax - 1; ++j) {
				const double dth  = t[j+1] - t[j];
				const double dt2  = dth*dth;
				const double dti  = 1.0/dth;
				const double dt2i = 1.0/dt2;
				const double dt3i = dt2i*dti;

				dt_sav[j]   = dth;
				dt2_sav[j]  = dt2;
				dti_sav[j]  = dti;
				dt2i_sav[j] = dt2i;
				dt3i_sav[j] = dt3i;
			}

			// construct the temperature and density deltas and their inverses
			for (int i = 0; i < imax - 1; ++i) {
				const double dd   = d[i+1] - d[i];
				const double dd2  = dd*dd;
				const double ddi  = 1.0/dd;
				const double dd2i = 1.0/dd2;
				const double dd3i = dd2i*ddi;

				dd_sav[i]   = dd;
				dd2_sav[i]  = dd2;
				ddi_sav[i]  = ddi;
				dd2i_sav[i] = dd2i;
				dd3i_sav[i] = dd3i;
			}

			helm_table.close();
		};
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