#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

void print(MatrixXd m) {
    for (size_t r = 0; r < m.rows(); ++r) {
        for (size_t c = 0; c < m.cols(); ++c) {
            cout << m(r, c);
            if (c != m.cols() - 1) {
                cout << ",";
            }
        }
        cout << endl;
    }
}

int main() {
    Vector3d v(10, 20, 30);
    print(v);

    Matrix2d m;
    m << 2, 3, 1, 4;
    print(m);

    return 0;
}
