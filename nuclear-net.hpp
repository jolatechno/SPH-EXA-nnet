#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>

#include "net14-constants.hpp"

/*
photodesintegration_to_first_order  R_i,j, n_i,j

fusion_to_first_order               F_i,j,k, n_i,j,k, Y_i
	'-> M : dY/dt = M*Y

include_temp : dM/dT, theta, value_1, value_2, BE'_i: sum_i(BE'_i*dY_i/dt) + value_1*T = value_2*dT/dt
	'-> M' : d{T, Y}/dt = M'*{T, Y}

iterate_system M', theta, Y, T
	'-> D{T, Y} = Dt* M'*(T_in*(1 - theta) + T_out*theta)

	D{T, Y} = Dt* M'*(T_in*(1 - theta) + T_out*theta)
 	<=> D{T, Y} = Dt* M'*({T_in, Y_in} + theta*{DT, DY})
 	<=> (I - Dt*M'()*theta)*D{T, Y} = DT*M'*{T_in, Y_in}
 	<=> D{T, Y} = (I - Dt* M' *theta)^{-1} * DT* M' *{T_in, Y_in}

solve_system : compose_system(Y, T), Y, T:
 	DT = 0, DY = 0

 	while true:
 		M' = compose_system(Y + theta*DY, T + theta*DT)

		DT, DY -> iterate_system(M', Y, T)

		abs((DT - DT_prev)/T) < tol:
		 	T = T + DT, Y = Y + DT

	return T + DT, Y + DY
*/

namespace nnet {
	namespace utils {
		template<typename Float>
		Eigen::SparseMatrix<Float> sparsify(const Eigen::Matrix<Float, -1, -1> &Min, const Float epsilon=1e-16) {
			/* -------------------
			put a "sparsified" version of Min into Mout according to epsilon 
			------------------- */
			std::vector<Eigen::Triplet<Float>> coefs;

			for (int i = 0; i < Min.cols(); ++i)
				for (int j = 0; j < Min.rows(); ++j)
					if (std::abs(Min(i, j)) > epsilon)
						coefs.push_back(Eigen::Triplet<Float>(i, j, Min(i, j)));

			Eigen::SparseMatrix<Float> Mout(Min.cols(), Min.rows());
			Mout.setFromTriplets(coefs.begin(), coefs.end());
			return Mout;
		}
	}

	template<class FloatMatrix, class IntMatrix>
	FloatMatrix photodesintegration_to_first_order(const FloatMatrix &r, const IntMatrix &n) {
		/* -------------------
		simply add the diagonal desintegration terms to the desintegration rates if not included

		makes sure that the equation ends up being : dY/dt = r*Y
		------------------- */

		int dimension = r.cols();
		FloatMatrix M = FloatMatrix::Zero(dimension, dimension);

		for (int i = 0; i < dimension; ++i)
			for (int j = 0; j < dimension; ++j)
				if (j != i) {
					M(j, i) =  r(j, i)*n(j, i);
					M(i, i) -= r(j, i);
				}

		return M;
	}

	template<typename Float, class IntTensor, class vector>
	Eigen::Matrix<Float, -1, -1> fusion_to_first_order(const Eigen::Tensor<Float, 3> &f, const IntTensor &n, const vector &Y) {
		/* -------------------
		include fusion rate into desintegration rates for a given state Y
		------------------- */

		const int dimension = Y.size();
		Eigen::Matrix<Float, -1, -1> M = Eigen::Matrix<Float, -1, -1>::Zero(dimension, dimension);

		// add fusion rates
		for (int i = 0; i < dimension; ++i)
			for (int j = 0; j < dimension; ++j)
				if (i != j) {
					// add i + i -> j
					M(j, i) += f(j, i, i)*Y(i)*n(j, i, i);
					M(i, i) -= f(j, i, i)*Y(i)*2;

					// add i + k -> j
					for (int k = 0; k < dimension; ++k)
						if (i != k) {
							M(j, i) += (f(j, i, k)*n(j, i, k) + f(j, k, i)*n(j, k, i))*Y(k)/2;
							M(i, i) -= (f(j, i, k)			  + f(j, k, i)			 )*Y(k);
						}
				}

		return M;
	}

	template<class matrix, class vector, typename Float>
	matrix include_temp(const matrix &M, const matrix &dMdT, const Float value_1, const Float value_2, const vector &BE, const vector &Y) {
		/* -------------------
		add a row to M based on BE so that, d{T, Y}/dt = M'*{T, Y}

		value_1, value_2 : BE.dY/dt + value_2*T = value_1*dT/dt
					   <=> (M * Y).BE / value_1 + value_2 / value_1 * T = dT/dt
					   <=> (M.t * BE).Y / value_1 + value_2 / value_1 * T = dT/dt

		dMdT : dY/dt = (M + DT*dMdT)*Y
		   <=> dY/dt = (M + dt*dMdT*dT/dt)*Y
		   <=> dY/dt = (M + dt*dMdT*dT/dt)*Y
		------------------- */

		const int dimension = Y.size();
		matrix Mp(dimension + 1, dimension + 1);

		// insert M
		Mp(Eigen::seq(1, dimension), Eigen::seq(1, dimension)) = M;
		Mp(0, 0) = value_2/value_1;

		// insert Y -> temperature terms
		Mp(0, Eigen::seq(1, dimension)) = -M.transpose()*BE/value_1;

		// insert temperature -> Y terms
		// Mp(Eigen::seq(1, dimension), 0) = dMdT*Y;

		return Mp;
	}

	template<class matrix, class vector, typename Float>
	vector solve_first_order(const vector &Y, const Float T, const matrix &Mp, const Float dt, const Float theta=1, const Float epsilon=1e-16) {
		/* -------------------
		Solves d{Y, T}/dt = RQ*Y using eigen:

		D{T, Y} = Dt* M'*(T_in*(1 - theta) + T_out*theta)
 	<=> D{T, Y} = Dt* M'*({T_in, Y_in} + theta*{DT, DY})
 	<=> (I - Dt*M'()*theta)*D{T, Y} = DT*M'*{T_in, Y_in}
		------------------- */

		const int dimension = Y.size();

		// construct vector
		vector Y_T(dimension + 1);
		Y_T << T, Y;

		// right hand side
		const vector RHS = Mp*Y_T*dt;

		// construct M
		matrix M = -theta*dt*Mp + Eigen::Matrix<Float, -1, -1>::Identity(dimension + 1, dimension + 1);

		// sparcify M
		auto sparse_M = utils::sparsify(M, epsilon);

		// now solve {Dy, DT}*M = RHS
		Eigen::BiCGSTAB<Eigen::SparseMatrix<Float>>  BCGST;
		BCGST.compute(sparse_M);
		auto const DY_T = BCGST.solve(RHS);

		// add to solution
		return DY_T;
	}

	template<class problem, class vector, typename Float>
	vector solve_system(const problem construct_system, const vector &Y, const Float T, const Float dt, const Float theta=1, const Float tol=1e-5, const Float epsilon=1e-16) {
		const int dimension = Y.size();

		// construct vector
		vector Y_T(dimension + 1), DY_T(dimension + 1), prev_DY_T(dimension + 1);
		Y_T << T, Y;

		auto M = construct_system(Y, T);
		prev_DY_T = solve_first_order(Y, T, M, dt, theta, epsilon);

		while (true) {
			// construct system
			auto next_Y = Y + theta*prev_DY_T(Eigen::seq(1, dimension));
			auto next_T = T + theta*prev_DY_T(0);
			M = construct_system(next_Y, next_T);

			// solve system
			DY_T = solve_first_order(Y, T, M, dt, theta, epsilon);

			// exit on condition
			if (std::abs(prev_DY_T(0) - DY_T(0))/(T + theta*DY_T(0)) < tol)
				return DY_T;

			prev_DY_T = DY_T;
		}
	}


	namespace net14 {
		/// constant mass-excendent values
		const Eigen::VectorXd BE = [](){
				Eigen::VectorXd BE_(14);
				BE_ << 0.0, 7.27440, 14.43580, 19.16680, 28.48280, 38.46680, 45.41480, 52.05380, 59.09380, 64.22080, 71.91280, 79.85180, 87.84680, 90.55480;
				return BE_;
			}();

		/// constant number of particle created by photodesintegrations
		const Eigen::Matrix<int, 14, 14> n_photodesintegrations = [](){
				Eigen::Matrix<int, 14, 14> n = Eigen::Matrix<int, 14, 14>::Zero();

				// C -> 3He
				n(0, 1) = 3;

				// Z <-> Z "+ 1"
				for (int i = 1; i < 13; ++i) {
					n(i, i + 1) = 1;
					n(i + 1, i) = 1;
				}

				return n;
			}();

		/// constant number of particle created by fusions
		const Eigen::Tensor<int, 3> n_fusions = [](){
				Eigen::Tensor<int, 3> n(14, 14, 14);
				n.setZero();

				// C + C -> Ne + He
				n(3, 1, 1) = 2;
				n(0, 1, 1) = 2;

				// C + O -> Mg + He
				n(4, 1, 2) = 2;
				n(0, 1, 2) = 2;

				// O + O -> Si + He
				n(5, 2, 2) = 2;
				n(0, 2, 2) = 2;

				// 3He -> C ????
				
				return n;
			}();


		/// function computing the coulombian correction
		template<typename Float>
		Eigen::Vector<Float, 14> ideal_gaz_correction(const Float T) {
			/* -------------------
			simply copute the coulombian correction of BE within net-14
			------------------- */
			Eigen::Vector<Float, 14> BE_corr(14);
			BE_corr = 3./2. * constants::Kb * constants::Na * T;

			return BE_corr;
		}

		/// function computing the coulombian correction
		template<class vector, typename Float>
		vector coulomb_correction(const vector &Y, const Float T, const Float rho) {
			/* -------------------
			simply copute the coulombian correction of BE within net-14
			------------------- */
			vector BE_corr(14);

			BE_corr(0) = 0;
			for (int i = 1; i < 14; ++i) {
				BE_corr(i) = 2.27e5 * std::pow((Float)constants::Z(i), 5./3.) * std::pow(rho * Y(i), 1./3.) / T;
			}

			return BE_corr;
		}

		/// 
		template<typename Float>
		Eigen::Matrix<Float, 14, 14> get_net14_desintegration_rates(const Float T) {
			/* -------------------
			simply copute desintegration rates within net-14
			------------------- */

			Eigen::Matrix<Float, 14, 14> r = Eigen::Matrix<Float, 14, 14>::Zero();

			// Z -> Z "+ 1"
			for (int i = 3; i < 13; ++i)
				r(i + 1, i) = std::exp(0); 

			// Z <- Z "+ 1"
			for (int i = 4; i < 14; ++i) {
				int k = get_temperature_range(T);
				r(i - 1, i) = constants::fits::choose[i][k]/constants::fits::choose[i + 1][k]*std::exp(
					0); 
			}

			return r;
		}

		template<typename Float>
		Eigen::Tensor<Float, 3> get_net14_fusion_rates(const Float T) {
			/* -------------------
			simply copute fusion rates within net-14
			------------------- */

			Eigen::Tensor<Float, 3> f(14, 14, 14);
			f.setZero();

			// TODO

			return f;
		}
	}
}