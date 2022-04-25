#pragma once

// avoid aproximations
#define EIGEN_FAST_MATH 0

#include <Eigen/Dense>
#include <Eigen/Sparse>

// solvers
#include <Eigen/SparseLU>
// #include<Eigen/SparseCholesky>
#ifndef SOLVER
	// #define ITERATIVE_SOLVER _
	// #define SOLVER Eigen::BiCGSTAB<Eigen::SparseMatrix<Float>>

	#define SOLVER Eigen::SparseLU<Eigen::SparseMatrix<Float>, Eigen::COLAMDOrdering<int>>

	// #define SOLVER Eigen::SimplicialL<Eigen::SparseMatrix<Float>>
#endif

namespace eigen {
	/// custom analytical solver
	template<typename Float>
	Eigen::Vector<Float, -1> solve(Eigen::Matrix<Float, -1, -1> &M, Eigen::Vector<Float, -1> &RHS) {
		const int dimension = RHS.size();
		Eigen::Vector<Float, -1> X(dimension);

		// reduce into upper triangular
		for (int i = 0; i < dimension; ++i) {
			for (int j = 0; j < i; ++j) {
				// include the jth line
				Float weight = M(i, j);
				M(i, j) = 0;

				RHS(i) -= weight*RHS(j);
				for (int k = j + 1; k < dimension; ++k)
					M(i, k) -= weight*M(j, k);
			}

			// normalize ith line
			Float diagonal = M(i, i);
			M(i, i) = 1;

			// normalize
			RHS[i] /= diagonal;
			for (int j = i + 1; j < dimension; ++j)
				M(i, j) /= diagonal;
		}

		// "back propagate" to solve
		for (int i = dimension - 1; i >= 0; --i) {
			Float res = RHS(i);

			for (int j = i + 1; j < dimension; ++j)
				res -= M(i, j)*X(j);

			X(i) = res;
		}

		return X;
	}
}