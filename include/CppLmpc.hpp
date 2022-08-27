#include <Eigen/Core>
#include "CppControl.hpp"

using namespace Eigen;

namespace CppLmpc{

    class CppLmpc
    {
    public:
        CppLmpc();
        ~CppLmpc();
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