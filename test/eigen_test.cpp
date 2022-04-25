#include <iostream>

#include "../src/eigen.hpp"

int main() {
	Eigen::VectorXd RHS(3);
	RHS << 1, -1, 2;

	Eigen::MatrixXd M(3, 3);
	M << 1, 2, -1,
		 2, 3, -1,
		 3, -2, 1;

	Eigen::MatrixXd M_copy = M;

	Eigen::VectorXd X = eigen::solve(M, RHS);

	std::cout << "M=\n" << M_copy << "\n";
	std::cout << "X=" << X.transpose() << "\n";
	std::cout << "M*X=" << (M_copy*X).transpose() << "\n";

	return 0;
}