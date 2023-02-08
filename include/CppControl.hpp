#pragma once

#include <Eigen/Core>
using namespace Eigen;

namespace CppLmpc{

    class CppControl final
    {
    public:
        CppControl() = delete;
        ~CppControl() = delete;
        static void c2d(const MatrixXd& Amat, const VectorXd& Bmat, double sampling_time, MatrixXd& Amat_d, MatrixXd& Bmat_d);
        static int factorial(int n);
        static MatrixXd discretizeAmat(const MatrixXd& Amat, double sampling_time, int order_of_taylor=10);
        static VectorXd discretizeBmat(const MatrixXd& Amat, const VectorXd Bmat, double sampling_time, int order_of_taylor=10, int division=1000);

    };
}