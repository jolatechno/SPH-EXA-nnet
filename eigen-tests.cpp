#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef Eigen::Triplet<double> eigen_triplet;

int main() {
	/* -----------------------------------------------------------
	-----------------------------------------------------------
	initialize dense matrix
	-----------------------------------------------------------
	----------------------------------------------------------- */
	Eigen::MatrixXd m0(2,3);
	m0(0,0) = 3;
	m0(1,0) = 2.5;
	m0(0,1) = -1;
	m0(1,1) = m0(1,0) + m0(0,1);

	std::cout << "m0=" << std::endl << m0 << std::endl << std::endl;



	/* -----------------------------------------------------------
	-----------------------------------------------------------
	initialize and use dense matrix
	-----------------------------------------------------------
	----------------------------------------------------------- */
	Eigen::VectorXd v(3);
	v(0) = 1;
	v(1) = 2;
	v(2) = 3;

	std::cout << "v=" << std::endl << v << std::endl << std::endl;
	std::cout << "m0.size()=" << m0.size() << "=" << m0.cols() << "*" << m0.rows() << ", v.size()=" << v.size() << std::endl;

	Eigen::MatrixXd m(3,3);
	m << m0, Eigen::VectorXd(3).transpose();

	std::cout << "m=concat(m, 0)=" << std::endl << m << std::endl << std::endl;

	m *= 0.8;
	m += Eigen::MatrixXd::Identity(3, 3);

	std::cout << "m=m*0.8 + I=" << std::endl << m << std::endl << std::endl;
	std::cout << "m*v=" << std::endl << m * v << std::endl << std::endl;



	/* -----------------------------------------------------------
	-----------------------------------------------------------
	initialize a sparse matrix
	-----------------------------------------------------------
	----------------------------------------------------------- */
	std::vector<eigen_triplet> coefs;
	coefs.push_back(eigen_triplet(0,0,1.2));
	coefs.push_back(eigen_triplet(1,1,-0.5));
	coefs.push_back(eigen_triplet(1,2,0.5));
	coefs.push_back(eigen_triplet(2,1,1.3));
	coefs.push_back(eigen_triplet(2,2,-1));

	Eigen::SparseMatrix<double> sparse_m(3,3), sparse_I(3,3);
   	sparse_m.setFromTriplets(coefs.begin(), coefs.end());

   	std::cout << "sparse_m=" << std::endl << sparse_m << std::endl << std::endl;

   	sparse_I.setIdentity();
   	sparse_m *= 0.8;
	sparse_m += sparse_I;

   	std::cout << "sparse_m=sparse_m*0.8 + I" << std::endl << sparse_m << std::endl << std::endl;


	/* -----------------------------------------------------------
	-----------------------------------------------------------
	first solver
	-----------------------------------------------------------
	----------------------------------------------------------- */
	Eigen::BiCGSTAB<Eigen::SparseMatrix<double>>  BCGST;
	BCGST.compute(sparse_m);
	Eigen::VectorXd x = BCGST.solve(v);

	std::cout << "sparse_m*x=v, x= " << std::endl << x << std::endl << std::endl;
	std::cout << "sparse_m*x=" << std::endl << sparse_m * x << std::endl;
}