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