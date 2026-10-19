// Standalone regression test for the SO(3) helpers of the Modern Robotics
// C++ library (mr::VecToso3, mr::MatrixExp3, mr::MatrixLog3).
//
// Verified invariants, for a deterministic unit axis `w` and angle theta in
// (0, pi/2]:
//   * VecToso3(w) is skew-symmetric.
//   * R = MatrixExp3(VecToso3(w) * theta) is orthogonal (R*R^T == I).
//   * det(R) == +1.
//   * MatrixLog3 inverts MatrixExp3: MatrixExp3(MatrixLog3(R)) == R.
//   * MatrixLog3 recovers the original so(3) element: MatrixLog3(R) == so3.
//
// Only claims behavior for angles strictly inside (0, pi/2]; near-zero and
// near-pi robustness are not asserted.

#include "modern_robotics.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace {

const double kPi = 3.14159265358979323846;
const double kTol = 1e-10;

int g_failures = 0;

void fail(const std::string& msg) {
  std::printf("  FAIL: %s\n", msg.c_str());
  ++g_failures;
}

void expectNear(double actual, double expected, const char* what,
                const std::string& casename) {
  const double err = std::abs(actual - expected);
  if (err > kTol) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "[%s] %s: actual %.17g, expected %.17g, |err| %.3g > tol %.3g",
                  casename.c_str(), what, actual, expected, err, kTol);
    fail(buf);
  }
}

void runCase(const std::string& name, const Eigen::Vector3d& axis,
             double angle) {
  const Eigen::Matrix3d so3 = mr::VecToso3(axis) * angle;

  // mr::VecToso3 must produce a skew-symmetric matrix.
  const Eigen::Matrix3d skew = mr::VecToso3(axis);
  expectNear((skew + skew.transpose()).norm(), 0.0,
             "VecToso3 skew-symmetry", name);

  const Eigen::Matrix3d R = mr::MatrixExp3(so3);

  // R must be orthogonal with determinant +1.
  expectNear((R * R.transpose() - Eigen::Matrix3d::Identity()).norm(), 0.0,
             "orthogonality |R*R^T - I|", name);
  expectNear(R.determinant(), 1.0, "det(R)", name);

  // Roundtrip: exp(log(R)) must reproduce R.
  const Eigen::Matrix3d logR = mr::MatrixLog3(R);
  const Eigen::Matrix3d R2 = mr::MatrixExp3(logR);
  expectNear((R2 - R).norm(), 0.0, "MatrixExp3(MatrixLog3(R)) - R", name);

  // MatrixLog3 must recover the original so(3) element for theta in (0, pi/2].
  expectNear((logR - so3).norm(), 0.0, "MatrixLog3(R) - so3", name);
}

}  // namespace

int main() {
  const Eigen::Vector3d axes[3] = {
      Eigen::Vector3d(1.0, 0.0, 0.0),
      Eigen::Vector3d(0.0, 0.0, 1.0),
      Eigen::Vector3d(1.0, 2.0, 3.0).normalized(),
  };
  const char* axis_names[3] = {"x", "z", "(1,2,3)/|(1,2,3)|"};
  const double angles[3] = {0.3, 0.9, kPi / 2.0};

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      char name[128];
      std::snprintf(name, sizeof(name), "axis=%s, theta=%.6g", axis_names[i],
                    angles[j]);
      runCase(name, axes[i], angles[j]);
    }
  }

  if (g_failures == 0) {
    std::printf("PASS: all SO(3) roundtrip invariants within tolerance %.3g\n",
                kTol);
    return 0;
  }
  std::printf("FAIL: %d invariant violation(s)\n", g_failures);
  return 1;
}
