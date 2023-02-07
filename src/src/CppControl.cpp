#include "CppControl.hpp"

using namespace Eigen;

namespace CppLmpc{

    int CppControl::factorial(int n){
        if(n <= 1)
            return 1;
        else
            return n * factorial(n-1);
    }

    void CppControl::c2d(const MatrixXd& Amat, const VectorXd& Bmat, double sampling_time, MatrixXd& Amat_d, MatrixXd& Bmat_d)
    {
        Amat_d = discretizeAmat(Amat, sampling_time);
        Bmat_d = discretizeBmat(Amat, Bmat, sampling_time);
    }

    MatrixXd CppControl::discretizeAmat(const MatrixXd& Amat, double sampling_time, int order_of_taylor){

        const int dim_of_system = Amat.cols();
        MatrixXd tmp = MatrixXd::Zero(dim_of_system, dim_of_system);
        for(int i = 0; i < order_of_taylor; i++){
            MatrixXd Amat_i_times = MatrixXd::Identity(dim_of_system, dim_of_system);
            double sampling_time_i_times = 1.0;
            for(auto j = 0; j < i; j++){
                Amat_i_times = Amat_i_times * Amat;
                sampling_time_i_times *= sampling_time;
            }

            tmp = tmp + Amat_i_times * sampling_time_i_times / factorial(i);
        }
        return tmp;
    }

    VectorXd CppControl::discretizeBmat(const MatrixXd& Amat, const VectorXd Bmat, double sampling_time, int order_of_taylor, int division){

        const int dim_of_system = Amat.cols();
        const double step = (double)(sampling_time / division);
        MatrixXd tmp = MatrixXd::Zero(dim_of_system, dim_of_system);
        for(auto i = 0; i < division; i++){
            double t = i * step;
            tmp = tmp + discretizeAmat(Amat, t, order_of_taylor) * step;
        }
        return tmp * Bmat;
    }

}