#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>

#include "net14.hpp"

/*
photodesintegration_to_first_order  R_i,j, n_i,j

fusion_to_first_order               F_i,j,k, n_i,j,k, Y_i
	'-> M : dY/dt = M*Y

include_temp : theta, value_1, cv, BE'_i: sum_i(BE'_i*dY_i/dt) + value_1*T = cv*dT/dt
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

/* class reaction: std::vector<std::pair<int, int>> reactant -> std::vector<std::pair<int, int>> product */
/* function "reaction_to_first_order": std:vector<std::pair<reaction, double>> -> Eigen::matrix */



namespace nnet {
	class reaction {
		std::vector<std::pair<int, int>> reactants;
		std::vector<std::pair<int, int>> products;
	};

	namespace utils {
		template<typename Float, int n>
		Eigen::SparseMatrix<Float> sparsify(const Eigen::Matrix<Float, n, n> &Min, const Float epsilon=1e-16) {
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

	template<typename Float, class vector>
	Eigen::Matrix<Float, -1, -1> first_order_from_reactions(const std::vector<std::pair<reaction, Float>> &reactions, vector const &Y) {
		/* -------------------
		reactes a sparce matrix M such that dY/dt = M*Y*
		from a list of reactions
		------------------- */
		const int dimension = Y.size();

		Eigen::Matrix<Float, -1, -1> M = Eigen::Matrix<Float, -1, -1>::Zero(dimension, dimension);

		/* -----------------------------------------
		TODO
		----------------------------------------- */

		return M;
	}

	template<class vector, typename Float>
	std::tuple<vector, Float> add_and_cleanup(const vector &Y, const Float T, const vector &DY_T, const Float epsilon=1e-10) {
		const int dimension = Y.size();

		Float next_T = T + DY_T(0);
		vector next_Y = Y + DY_T(Eigen::seq(1, dimension));

		for (int i = 0; i < dimension; ++i)
			if (next_Y(i) < epsilon)
				next_Y(i) = 0;

		return {next_Y, next_T};
	}

	template<class matrix, class vector, typename Float>
	matrix include_temp(const matrix &M, const Float value_1, const Float cv, const vector &BE, const vector &Y) {
		/* -------------------
		add a row to M based on BE so that, d{T, Y}/dt = M'*{T, Y}

		value_1, cv : BE.dY/dt + value_1*T = cv*dT/dt
					   <=> (M * Y).BE / cv + value_1/ cv * T = dT/dt
					   <=> (M.t * BE).Y / cv + value_1 / cv * T = dT/dt
		------------------- */

		const int dimension = Y.size();
		matrix Mp = matrix::Zero(dimension + 1, dimension + 1);

		// insert M
		Mp(Eigen::seq(1, dimension), Eigen::seq(1, dimension)) = M;
		Mp(0, 0) = value_1/cv;

		// insert Y -> temperature terms
		Mp(0, Eigen::seq(1, dimension)) = M.transpose()*BE/cv;

		return Mp;
	}

	template<class matrix, class vector, typename Float>
	vector solve_first_order(const vector &Y, const Float T, const matrix &Mp, const Float dt, const Float theta=1, const Float epsilon=1e-16) {
		/* -------------------
		Solves d{Y, T}/dt = RQ*Y using eigen:

		D{T, Y} = Dt* M'*{T_in + theta*DT, Y_in + theta*DY}
 	<=> D{T, Y} = Dt* M'*({T_in, Y_in} + theta*D{T, Y})
 	<=> (I - Dt*M'*theta)*D{T, Y} = Dt*M'*{T_in, Y_in}
		------------------- */

		const int dimension = Y.size();

		// construct vector
		vector Y_T(dimension + 1);
		Y_T << T, Y;

		// right hand side
		const vector RHS = Mp*Y_T*dt;

		// construct M
		matrix M = -theta*dt*Mp + matrix::Identity(dimension + 1, dimension + 1);

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
		vector DY_T(dimension + 1), prev_DY_T(dimension + 1);

		{
			auto M = construct_system(Y, T);
			prev_DY_T = solve_first_order(Y, T, M, dt, theta, epsilon);
		}

		int max_iter = std::max(0., -std::log2(tol));
		for (int i = 0;; ++i) {
			// construct system
			vector scaled_DY_T = theta*prev_DY_T;
			auto [next_Y, next_T] = add_and_cleanup(Y, T, scaled_DY_T, epsilon);
			auto M = construct_system(Y, T);

			// solve system
			DY_T = solve_first_order(Y, T, M, dt, theta, epsilon);

			// exit on condition
			if (std::abs(prev_DY_T(0) - DY_T(0))/(T + theta*DY_T(0)) < tol || i == max_iter)
				return DY_T;

			prev_DY_T = DY_T;
		}
	}































	/* -----------------------------------------
	--------------------------------------------
	useless from now on
	--------------------------------------------
	----------------------------------------- */

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
}