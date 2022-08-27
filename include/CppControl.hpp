#include <Eigen/Core>
using namespace Eigen;

namespace CppLmpc{

    class CppControl
    {
    public:
        CppControl();
        ~CppControl();
        void c2d(const MatrixXd& Amat, const VectorXd& Bmat, double sampling_time, MatrixXd& Amat_d, MatrixXd& Bmat_d);
        int factorial(int n);
        MatrixXd discretizeAmat(const MatrixXd& Amat, double sampling_time, int order_of_taylor=10);
        VectorXd discretizeBmat(const MatrixXd& Amat, const VectorXd Bmat, double sampling_time, int order_of_taylor=10, int division=1000);

    };
}