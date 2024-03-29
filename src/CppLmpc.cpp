#include "CppLmpc.hpp"

using namespace Eigen;

namespace CppLmpc{

    CppLmpc::CppLmpc(double sampling_time, const MatrixXd& Ac, const MatrixXd& Bc, const MatrixXd& Cc, const MatrixXd& Dc, int Hp, int Hu, int Hw)
    : sampling_time_(sampling_time),
    Ac_(Ac),
    Bc_(Bc),
    Cc_(Cc),
    Dc_(Dc),
    Hp_(Hp),
    Hu_(Hu),
    Hw_(Hw)
    {
        MatrixXd Ad;
        MatrixXd Bd;
        MatrixXd Cd = Cc_;
        MatrixXd Dd = Dc_;
        CppControl::c2d(Ac, Bc_, sampling_time_, Ad, Bd);

        Ap_ = calcAp(Ad, Hp_, Hw_);
        Bp1_ = calcBp1(Ad, Bd, Hp_, Hw_, Hu_);
        Bp2_ = calcBp2(Ad, Bd, Hp_, Hw_, Hu_);
        Cp_ = calcCp(Ad, Bd, Cd, Hp_, Hw_);

        // Phi, Psi, Theta
        Phi_ = Cp_ * Ap_;
        Psi_ = Cp_ * Bp1_;
        Theta_ = Cp_ * Bp2_;

        W_ = calcInputRateIneqMat(std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), Hu_);
        w_ = calcInputRateIneqVec(std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), Hu_);

        F_ = calcInputIneqMatF(std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), Hu_);
        F1_ = calcInputIneqMatF1(std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), Hu_);
        f_ = calcInputIneqVec(std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), Hu_);

        Gamma_ = calcOutputIneqMat(std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), Hp_);
        gamma_ = calcOutputIneqVec(std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), Hp_);

        int xdim = Bd.rows();
        H_ = calcStateIneqMat(VectorXd::Ones(xdim) * std::numeric_limits<double>::max(), VectorXd::Ones(xdim) * std::numeric_limits<double>::min(), xdim, Hp_);
        eta_ = calcStateIneqVec(VectorXd::Ones(xdim) * std::numeric_limits<double>::max(), VectorXd::Ones(xdim) * std::numeric_limits<double>::min(), xdim, Hp_);

        Q_ = MatrixXd::Identity(Hp_-Hw_, Hp_-Hw_);
        R_ = MatrixXd::Identity(Hu_, Hu_);
        S_ = MatrixXd::Identity(Hu_ * Bd.rows(), Hu_ * Bd.rows());

        Tau_ = MatrixXd::Zero(Hp_-Hw_, 1); // Tr
        Bstate_ = Bp2_;
        Hess_ = Theta_.transpose() * Q_ * Theta_ + Bstate_.transpose() * S_ * Bstate_ + R_;
        Gineq_ = Gamma_ * Theta_;
        Hineq_ = H_ * Bp2_;
    };

    CppLmpc::~CppLmpc(){};

    MatrixXd CppLmpc::pow(const MatrixXd& A, int n){
        MatrixXd Apow = A;
        for(auto i=0; i<n-1; i++){
            Apow = Apow * A;
        }
        return Apow;
    }

    MatrixXd CppLmpc::vstack(const MatrixXd& A, const MatrixXd& B){
        int col = A.cols();
        if ( B.cols() > A.cols() ){
            col = B.cols();
        }
        MatrixXd C( A.rows() + B.rows(), col);
        for ( auto i=0; i<A.rows(); i++ ){
            for ( auto j=0; j<A.cols(); j++ ){
                C( i, j ) = A( i, j );
            }
        }
        for ( auto i=0; i<B.rows(); i++ ){
            for ( int j=0; j<B.cols(); j++ ){
                C( A.rows() + i, j ) = B( i, j );
            }
        }
        return C;
    }

    MatrixXd CppLmpc::hstack(const MatrixXd& A, const MatrixXd& B){
        int row = A.rows();
        if ( B.rows() > A.rows() ){
            row = B.rows();
        }
        MatrixXd C( row, A.cols() + B.cols());
        for ( auto i=0; i<A.rows(); i++ ){
            for ( auto j=0; j<A.cols(); j++ ){
                C( i, j ) = A( i, j );
            }
        }
        for ( auto i=0; i<B.rows(); i++ ){
            for ( auto j=0; j<B.cols(); j++ ){
                C( i, A.cols() + j ) = B( i, j );
            }
        }
        return C;
    }

    // calculate system matrix Ap of predictive model
    MatrixXd CppLmpc::calcAp(const MatrixXd& Ad, int Hp, int Hw){
        MatrixXd Ap;
        for (auto i=0; i<Hp-Hw; i++){
            if(i == 0) Ap=Ad;
            else {
                Ap = vstack(Ap, pow(Ad, i+1));
            }
        }
        return Ap;
    }

    // calculate system matrix Bp1 of predictive model
    MatrixXd CppLmpc::calcBp1(const MatrixXd& Ad, const MatrixXd& Bd, int Hp, int Hw, int Hu){

        // Bp1
        MatrixXd Bp1;
        MatrixXd tmp;
        for (auto i=0; i<Hp-Hw; i++){
            tmp = MatrixXd::Zero(Ad.rows(), 1);
            if (i == 0){
                Bp1 = Bd;
            }
            else{
                for (auto j=0; j<i+1; j++){
                    tmp = tmp + pow(Ad, j) * Bd;
                }
                Bp1 = vstack(Bp1, tmp);
            }
        }
        return Bp1;
    }

    // calculate system matrix Bp2 of predictive model
    MatrixXd CppLmpc::calcBp2(const MatrixXd& Ad, const MatrixXd& Bd, int Hp, int Hw, int Hu){

        // Bp2
        MatrixXd Bp2;

        // Bp21
        MatrixXd tmp1;
        MatrixXd tmp2;
        MatrixXd Bp21;
        MatrixXd Bp22;
        MatrixXd contenor_for_zero;
        contenor_for_zero = MatrixXd::Zero(Ad.rows(), 1);
        for (auto i=0; i<Hu; i++){
            tmp1 = MatrixXd::Zero(Ad.rows(), 1);
            if (i == 0){
                tmp1 = Bd;
            }
            else{
                for (auto j=0;  j<i-1; j++){
                    tmp1 = vstack(tmp1, contenor_for_zero);
                }
                tmp1 = vstack(tmp1, Bd);
            }
            tmp2 = Bd;
            for (auto j=0; j<Hu-1-i; j++){
                for(auto k = 0; k<j+1; k++){
                    tmp2 = tmp2 + pow(Ad, k+1) * Bd;
                }
                tmp1 = vstack(tmp1, tmp2);
            }
            if (i == 0){
                Bp21 = tmp1;
            }
            else{
                Bp21 = hstack(Bp21, tmp1);
            }
        }

        // Bp22
        if(Hp-Hw-Hu>0){
            for (int i = 0;  i < Hp - Hw - Hu; i++){// i = 0:Hp-Hu
                for (int j = 0;  j < Hu; j++){// i = 0:Hu
                    tmp1 = MatrixXd::Zero(Ad.rows(), 1);
                    for (int k = 0;  k < Hu + 1 - j; k++){// k = 0:Hu
                        tmp1 = tmp1 + pow(Ad, k) * Bd;
                    }
                    if (j == 0){
                        tmp2 = tmp1;
                    }
                    else{
                        tmp2 = hstack(tmp2, tmp1);
                    }
                }
                if (i == 0){
                    Bp22 = tmp2;
                }
                else{
                    Bp22 = vstack(Bp22, tmp2);
                }
                Bp2 = vstack(Bp21, Bp22);
            }
        }
        else{
            Bp2 = Bp21;
        }

        return Bp2;
    }

    // Cp
    MatrixXd CppLmpc::calcCp(const MatrixXd& Ad, const MatrixXd& Bd, const MatrixXd& Cd, int Hp, int Hw){
        MatrixXd tmp;
        MatrixXd Cp;
        MatrixXd contenor_for_zero = MatrixXd::Zero(1, Ad.rows());
        for (auto i=0;  i<Hp-Hw; i++){// i = 0:Hp-Hw
            for (auto j=0;  j<Hp-Hw; j++){// i = 0:Hp-Hw
                if (j == 0){
                    if (i == 0){
                        tmp = Cd;
                    }
                    else{
                        tmp = contenor_for_zero;
                    }
                }
                else{
                    if (i == j){
                        tmp = hstack(tmp, Cd);
                    }
                    else{
                        tmp = hstack(tmp, contenor_for_zero);
                    }
                }
            }
            if (i == 0){
                Cp = tmp;
            }
            else{
                Cp = vstack(Cp, tmp);
            }
        }
        return Cp;
    }

    MatrixXd CppLmpc::calcInputIneqMatF(double u_max, double u_min, int Hu){
        // For a scalar input constraints
        //F * u <= f
        //F = [F1 F2 ... Fhu]
        //=> F' * delta_u <= -F1' * u[k-1|k] - f
        MatrixXd F = MatrixXd::Zero(Hu * 2, Hu);
        for (int i = 0;  i < Hu; i++){// i = 0:Hp-Hw
            for (int j = 0;  j < i + 1; j++){// i = 0:Hp-Hw
                F(i * 2, j) = 1.0;
                F(i * 2 + 1, j) = -1.0;
            }
        }
        return F;
    }

    MatrixXd CppLmpc::calcInputIneqMatF1(double u_max, double u_min, int Hu){
        // For a scalar input constraints
        //F * u <= 0
        //F = [F1 F2 ... Fhu]
        //=> F' * delta_u <= -F1' * u[k-1|k] - f
        MatrixXd F1 = MatrixXd::Zero(Hu * 2, 1);
        for (int i = 0;  i < Hu; i++){// i = 0:Hp-Hw
            F1(i * 2) = 1.0;
            F1(i * 2 + 1) = -1.0;
        }
        return F1;
    }

    MatrixXd CppLmpc::calcInputIneqVec(double u_max, double u_min, int Hu){
        // For a scalar input constraints
        //F * u <= 0
        //F = [F1 F2 ... Fhu],  F' = [F1' F2' ... Fhu']
        //=> F' * delta_u <= -F1' * u[k-1|k] - f
        MatrixXd f = MatrixXd::Zero(Hu * 2, 1);
        for (int i = 0;  i < Hu; i++){// i = 0:Hp-Hw
            f(i * 2) = -u_max;
            f(i * 2 + 1) = u_min;
        }
        return f;
    }

    MatrixXd calcInputRateIneqMat(double du_max, double du_min, int Hu){
        // For a scalar input rate constraints
        //W * u <= w
        //W = [W1 W2 ... Whu],  w : scalar value
        MatrixXd W = MatrixXd::Zero(Hu*2, Hu);
        for (int i = 0;  i < Hu; i++){// i = 0:Hp-Hw
            W(i * 2, i) = 1.0;
            W(i * 2 + 1, i) = -1.0;
        }
        return W;
    }

    MatrixXd calcInputRateIneqVec(double du_max, double du_min, int Hu){
        // For a scalar input rate constraints
        //W * u <= w
        //W = [W1 W2 ... Whu],  w : scalar value
        MatrixXd w = MatrixXd::Zero(Hu*2, 1);
        for (int i = 0;  i < Hu; i++){// i = 0:Hp-Hw
            w(i * 2) = du_max;
            w(i * 2 + 1) = -du_min;
        }
        return w;
    }

    MatrixXd CppLmpc::calcOutputIneqMat(double z_max, double z_min, int Hp){
        // For a scalar output constraints
        //Γ * Z[k] + g <= 0
        MatrixXd Gamma = MatrixXd::Zero(Hp*2, Hp);
        for (int i = 0;  i < Hp; i++){// i = 0:Hp-Hw
            Gamma(i * 2, i) = 1.0;
            Gamma(i * 2 + 1, i) = -1.0;
        }
        return Gamma;
    }

    MatrixXd CppLmpc::calcOutputIneqVec(double z_max, double z_min, int Hp){
        // For a scalar output constraints
        //Γ * Z[k] + g <= 0
        MatrixXd gamma = MatrixXd::Zero(Hp*2, 1);
        for (int i = 0;  i < Hp; i++){// i = 0:Hp-Hw
            gamma(i * 2) = -z_max;
            gamma(i * 2 + 1) = z_min;
        }
        return gamma;
    }

    MatrixXd CppLmpc::calcStateIneqMat(const VectorXd& x_max, const VectorXd& x_min, int xdim, int Hp){
        //H * X[k] + η <= 0
        //x_max = [x1_max x2_max ... xn_max]^T
        //x_min = [x1_min x2_min ... xn_min]^T
        MatrixXd H_p = MatrixXd::Zero(xdim*2, xdim);
        MatrixXd h = MatrixXd::Zero(xdim*2, 1);
        for (int i = 0;  i < xdim; i++){// i = 0:Hp-Hw
            H_p(i * 2, i) = 1.0;
            H_p(i * 2 + 1, i) = -1.0;
        }
        for (int i = 0;  i < xdim; i++){// i = 0:Hp-Hw
            h(i * 2) = -x_max(i);
            h(i * 2 + 1) = x_min(i);
        }

        //HP
        MatrixXd H_P;
        MatrixXd eta = MatrixXd::Zero(xdim*2, 1);
        MatrixXd tmp;
        MatrixXd contenor_for_zero = MatrixXd::Zero(xdim*2, xdim);
        for (int i = 0;  i < Hp; i++){// i = 0:Hp-Hw
            for (int j = 0;  j < Hp; j++){// i = 0:Hp-Hw
                if (j == 0){
                    if (i == 0){
                        tmp = H_p;
                    }
                    else{
                        tmp = contenor_for_zero;
                    }
                }
                else if (i == j){
                    tmp = hstack(tmp, H_p);
                }
                else{
                    tmp = hstack(tmp, contenor_for_zero);
                }
            }
            if (i == 0){
                H_P = tmp;
                eta = h;
            }
            else{
                H_P = vstack(H_P, tmp);
                eta = vstack(eta, h);
            }
        }
        return H_P;
    }

    MatrixXd CppLmpc::calcStateIneqVec(const VectorXd& x_max, const VectorXd& x_min, int xdim, int Hp){
        //H * X[k] + η <= 0
        //x_max = [x1_max x2_max ... xn_max]^T
        //x_min = [x1_min x2_min ... xn_min]^T
        MatrixXd H_p = MatrixXd::Zero(xdim*2, xdim);
        MatrixXd h = MatrixXd::Zero(xdim*2, 1);
        for (int i = 0;  i < xdim; i++){// i = 0:Hp-Hw
            H_p(i * 2, i) = 1.0;
            H_p(i * 2 + 1, i) = -1.0;
        }
        for (int i = 0;  i < xdim; i++){// i = 0:Hp-Hw
            h(i * 2) = -x_max(i);
            h(i * 2 + 1) = x_min(i);
        }

        //HP
        MatrixXd H_P;
        MatrixXd eta = MatrixXd::Zero(xdim*2, 1);
        MatrixXd tmp;
        MatrixXd contenor_for_zero = MatrixXd::Zero(xdim*2, xdim);
        for (int i = 0;  i < Hp; i++){// i = 0:Hp-Hw
            for (int j = 0;  j < Hp; j++){// i = 0:Hp-Hw
                if (j == 0){
                    if (i == 0){
                        tmp = H_p;
                    }
                    else{
                        tmp = contenor_for_zero;
                    }
                }
                else if (i == j){
                    tmp = hstack(tmp, H_p);
                }
                else{
                    tmp = hstack(tmp, contenor_for_zero);
                }
            }
            if (i == 0){
                H_P = tmp;
                eta = h;
            }
            else{
                H_P = vstack(H_P, tmp);
                eta = vstack(eta, h);
            }
        }
        return eta;
    }

}
