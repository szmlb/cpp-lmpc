#pragma once
#include <Eigen/Core>
#include "CppControl.hpp"

using namespace Eigen;

namespace CppLmpc{

    class CppLmpc
    {
    public:
        CppLmpc(double sampling_time, const MatrixXd& Ac, const MatrixXd& Bc, const MatrixXd& Cc, const MatrixXd& Dc);
        ~CppLmpc();
    private:
        double sampling_time_;
        MatrixXd Ac_;
        MatrixXd Bc_;
        MatrixXd Cc_;
        MatrixXd Dc_;

        MatrixXd Ap_; //= calcApForMPC(Ad, Hp, Hw);
        MatrixXd Bp1_; //= calcBp1ForMPC(Ad, Bd, Hp, Hw, Hu);
        MatrixXd Bp2_; //= calcBp2ForMPC(Ad, Bd, Hp, Hw, Hu);
        MatrixXd Cp_; //= calcCpForMPC(Ad, Bd, Cd, Hp, Hw);

        // Phi, Psi, Theta
        MatrixXd Phi_; // = Cp * Ap;
        MatrixXd Psi_; //= Cp * Bp1;
        MatrixXd Theta_; //= Cp * Bp2;

        MatrixXd W_; // = calcInputRateIneqMat_1(du_max, du_min, Hu);
        MatrixXd w_; // = calcInputRateIneqMat_2(du_max, du_min, Hu);

        MatrixXd Aineq_; // = calcInputIneqMat_1(u_max, u_min, Hu);
        MatrixXd Aineq1_; // = calcInputIneqMat_2(u_max, u_min, Hu);
        MatrixXd Aineq_bias_; // = calcInputIneqMat_3(u_max, u_min, Hu);

        MatrixXd Gamma_; // = calcOutputIneqMat_1(z_max, z_min, Hp);
        MatrixXd gamma_; // = calcOutputIneqMat_2(z_max, z_min, Hp);
        MatrixXd H_; // = calcStateIneqMat_1(x_max, x_min, Bd.rows(), Hp);
        MatrixXd eta_; // = calcStateIneqMat_2(x_max, x_min, Bd.rows(), Hp);

        MatrixXd Tau_; // = MatrixXd::Zero(Hp-Hw, 1); // Tr
        MatrixXd Bstate_; // = Bp2;
        MatrixXd Hess_; // = Theta.transpose() * Q * Theta + Bstate.transpose() * S * Bstate + R;
        MatrixXd Gineq_; // = Gamma * Theta;
        MatrixXd Hineq_; // = H * Bp2;

        MatrixXd pow(const MatrixXd& A, int n);
        MatrixXd vstack(const MatrixXd& A, const MatrixXd& B);
        MatrixXd hstack(const MatrixXd& A, const MatrixXd& B);
        MatrixXd calcAp(const MatrixXd& Ad, int Hp, int Hw);
        MatrixXd calcBp1(const MatrixXd& Ad, const MatrixXd& Bd, int Hp, int Hw, int Hu);
        MatrixXd calcBp2(const MatrixXd& Ad, const MatrixXd& Bd, int Hp, int Hw, int Hu);
        MatrixXd calcCp(const MatrixXd& Ad, const MatrixXd& Bd, const MatrixXd& Cd, int Hp, int Hw);
        MatrixXd calcInputIneqMatF(double u_max, double u_min, int Hu);
        MatrixXd calcInputIneqMatF1(double u_max, double u_min, int Hu);
        MatrixXd calcInputIneqVecf(double u_max, double u_min, int Hu);
        MatrixXd calcOutputIneqMat(double z_max, double z_min, int Hp);
        MatrixXd calcOutputIneqVec(double z_max, double z_min, int Hp);
        MatrixXd calcStateIneqMat(const MatrixXd& x_max, const MatrixXd& x_min, int xdim, int Hp);
        MatrixXd calcStateIneqVec(const MatrixXd& x_max, const MatrixXd& x_min, int xdim, int Hp);
    };

}