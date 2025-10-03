"""Tests for general operation definitions."""

from typing import Type

import mujoco
import numpy as np
from absl.testing import absltest, parameterized

from mink.exceptions import InvalidMocapBody
from mink.lie.base import MatrixLieGroup
from mink.lie.se3 import SE3, _getQ
from mink.lie.so3 import SO3

from .utils import assert_transforms_close


@parameterized.named_parameters(
    ("SO3", SO3),
    ("SE3", SE3),
)
class TestOperations(parameterized.TestCase):
    def test_inverse_bijective(self, group: Type[MatrixLieGroup]):
        """Check inverse of inverse."""
        transform = group.sample_uniform()
        assert_transforms_close(transform, transform.inverse().inverse())

    def test_matrix_bijective(self, group: Type[MatrixLieGroup]):
        """Check that we can convert to and from matrices."""
        transform = group.sample_uniform()
        assert_transforms_close(transform, group.from_matrix(transform.as_matrix()))

    def test_log_exp_bijective(self, group: Type[MatrixLieGroup]):
        """Check 1-to-1 mapping for log <=> exp operations."""
        transform = group.sample_uniform()

        tangent = transform.log()
        self.assertEqual(tangent.shape, (group.tangent_dim,))

        exp_transform = group.exp(tangent)
        assert_transforms_close(transform, exp_transform)
        np.testing.assert_allclose(tangent, exp_transform.log())

    def test_adjoint(self, group: Type[MatrixLieGroup]):
        transform = group.sample_uniform()
        omega = np.random.randn(group.tangent_dim)
        assert_transforms_close(
            transform @ group.exp(omega),
            group.exp(transform.adjoint() @ omega) @ transform,
        )

    def test_rminus(self, group: Type[MatrixLieGroup]):
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        T_c = T_a.inverse() @ T_b
        np.testing.assert_allclose(T_b.rminus(T_a), T_c.log())

    def test_lminus(self, group: Type[MatrixLieGroup]):
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        np.testing.assert_allclose(T_a.lminus(T_b), (T_a @ T_b.inverse()).log())

    def test_rplus(self, group: Type[MatrixLieGroup]):
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        T_c = T_a.inverse() @ T_b
        assert_transforms_close(T_a.rplus(T_c.log()), T_b)

    def test_lplus(self, group: Type[MatrixLieGroup]):
        T_a = group.sample_uniform()
        T_b = group.sample_uniform()
        T_c = T_a @ T_b.inverse()
        assert_transforms_close(T_b.lplus(T_c.log()), T_a)

    def test_jlog(self, group: Type[MatrixLieGroup]):
        state = group.sample_uniform()
        w = np.random.rand(state.tangent_dim) * 1e-4
        state_pert = state.plus(w).log()
        state_lin = state.log() + state.jlog() @ w
        np.testing.assert_allclose(state_pert, state_lin, atol=1e-7)


class TestGroupSpecificOperations(absltest.TestCase):
    """Group specific tests."""

    # SO3.

    def test_so3_equality(self):
        rot_1 = SO3.identity()
        rot_2 = SO3.identity()
        self.assertEqual(rot_1, rot_2)

        rot_1 = SO3.from_x_radians(np.pi)
        rot_2 = SO3.from_x_radians(np.pi)
        self.assertEqual(rot_1, rot_2)

        rot_1 = SO3.from_x_radians(np.pi)
        rot_2 = SO3.from_x_radians(np.pi * 0.5)
        self.assertNotEqual(rot_1, rot_2)

        # Make sure different types are properly handled.
        self.assertNotEqual(SO3.identity(), 5)

    def test_so3_rpy_bijective(self):
        T = SO3.sample_uniform()
        assert_transforms_close(T, SO3.from_rpy_radians(*T.as_rpy_radians()))

    def test_so3_raises_error_if_invalid_shape(self):
        with self.assertRaises(ValueError):
            SO3(wxyz=np.random.rand(2))

    def test_so3_copy(self):
        T = SO3.sample_uniform()
        T_c = T.copy()
        np.testing.assert_allclose(T_c.wxyz, T.wxyz)
        T.wxyz[0] = 1.0
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(T_c.wxyz, T.wxyz)

    def test_so3_interpolate(self):
        start = SO3.from_y_radians(np.pi)
        end = SO3.from_y_radians(2 * np.pi)

        assert_transforms_close(start.interpolate(end), SO3.from_y_radians(np.pi * 1.5))
        assert_transforms_close(
            start.interpolate(end, alpha=0.75), SO3.from_y_radians(np.pi * 1.75)
        )

        assert_transforms_close(start.interpolate(end, alpha=0.0), start)
        assert_transforms_close(start.interpolate(end, alpha=1.0), end)

        expected_error_message = "Expected alpha within [0.0, 1.0]"
        with self.assertRaises(ValueError) as cm:
            start.interpolate(end, alpha=-1.0)
        self.assertIn(expected_error_message, str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            start.interpolate(end, alpha=2.0)
        self.assertIn(expected_error_message, str(cm.exception))

    def test_so3_apply(self):
        vec = np.random.rand(3)
        rot = SO3.sample_uniform()
        rotated_vec = rot.apply(vec)
        np.testing.assert_allclose(rotated_vec, rot.as_matrix() @ vec)

    def test_so3_matmul_with_vector_calls_apply(self):
        """Using @ with an ndarray should delegate to .apply(...)."""
        vec = np.random.rand(3)
        rot = SO3.sample_uniform()
        np.testing.assert_allclose(rot @ vec, rot.apply(vec))

    def test_so3_apply_throws_assertion_error_if_wrong_shape(self):
        rot = SO3.sample_uniform()
        vec = np.random.rand(2)
        with self.assertRaises(AssertionError):
            rot.apply(vec)

    def test_so3_clamp(self):
        # Clamping with the default RPY limits (+- infinity) means the SO3 should remain unchanged.
        rot = SO3.from_rpy_radians(roll=np.pi, pitch=0, yaw=-np.pi)
        self.assertEqual(rot, rot.clamp())

        original_rpy = rot.as_rpy_radians()

        # Test clamping the roll.
        clamped_rot = rot.clamp(roll_radians=(0, 1))
        clamped_rpy = clamped_rot.as_rpy_radians()
        self.assertAlmostEqual(clamped_rpy.roll, 1)
        self.assertAlmostEqual(clamped_rpy.pitch, original_rpy.pitch)
        self.assertAlmostEqual(clamped_rpy.yaw, original_rpy.yaw)

        # Test clamping the pitch.
        clamped_rot = rot.clamp(pitch_radians=(1, 2))
        clamped_rpy = clamped_rot.as_rpy_radians()
        self.assertAlmostEqual(clamped_rpy.roll, original_rpy.roll)
        self.assertAlmostEqual(clamped_rpy.pitch, 1)
        self.assertAlmostEqual(clamped_rpy.yaw, original_rpy.yaw)

        # Test clamping the yaw.
        clamped_rot = rot.clamp(yaw_radians=(np.pi, 2 * np.pi))
        clamped_rpy = clamped_rot.as_rpy_radians()
        self.assertAlmostEqual(clamped_rpy.roll, original_rpy.roll)
        self.assertAlmostEqual(clamped_rpy.pitch, original_rpy.pitch)
        self.assertAlmostEqual(clamped_rpy.yaw, np.pi)

    # SE3.

    def test_se3_raises_error_if_invalid_shape(self):
        with self.assertRaises(ValueError):
            SE3(wxyz_xyz=np.random.rand(2))

    def test_se3_equality(self):
        pose_1 = SE3.identity()
        pose_2 = SE3.identity()
        self.assertEqual(pose_1, pose_2)

        pose_1 = SE3.from_translation(np.array([1.0, 0.0, 0.0]))
        pose_2 = SE3.from_translation(np.array([1.0, 0.0, 0.0]))
        self.assertEqual(pose_1, pose_2)

        pose_1 = SE3.from_translation(np.array([1.0, 2.0, 3.0]))
        pose_2 = SE3.from_translation(np.array([1.0, 0.0, 0.0]))
        self.assertNotEqual(pose_1, pose_2)

        # Make sure different types are properly handled.
        self.assertNotEqual(SE3.identity(), 5)

    def test_se3_apply(self):
        T = SE3.sample_uniform()
        v = np.random.rand(3)
        np.testing.assert_allclose(
            T.apply(v), T.as_matrix()[:3, :3] @ v + T.translation()
        )

    def test_se3_matmul_with_vector_calls_apply(self):
        """Using @ with an ndarray should delegate to .apply(...)."""
        vec = np.random.rand(3)
        T = SE3.sample_uniform()
        np.testing.assert_allclose(T @ vec, T.apply(vec))

    def test_se3_from_mocap_id(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body name="mocap" mocap="true" pos=".5 1 5" quat="1 1 0 0">
              <geom type="sphere" size=".1" mass=".1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        mid = model.body("mocap").mocapid[0]
        pose = SE3.from_mocap_id(data, mocap_id=mid)
        np.testing.assert_allclose(pose.translation(), data.mocap_pos[mid])
        np.testing.assert_allclose(pose.rotation().wxyz, data.mocap_quat[mid])

    def test_se3_from_mocap_name(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body name="mocap" mocap="true" pos=".5 1 5" quat="1 1 0 0">
              <geom type="sphere" size=".1" mass=".1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        pose = SE3.from_mocap_name(model, data, "mocap")
        mid = model.body("mocap").mocapid[0]
        np.testing.assert_allclose(pose.translation(), data.mocap_pos[mid])
        np.testing.assert_allclose(pose.rotation().wxyz, data.mocap_quat[mid])

    def test_se3_from_mocap_name_raises_error_if_body_not_mocap(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body name="test" pos=".5 1 5" quat="1 1 0 0">
              <geom type="sphere" size=".1" mass=".1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        with self.assertRaises(InvalidMocapBody):
            SE3.from_mocap_name(model, data, "test")

    def test_se3_interpolate(self):
        start = SE3.from_rotation_and_translation(
            SO3.from_x_radians(0.0), np.array([0, 0, 0])
        )
        end = SE3.from_rotation_and_translation(
            SO3.from_x_radians(np.pi), np.array([1, 0, 0])
        )

        assert_transforms_close(
            start.interpolate(end),
            SE3.from_rotation_and_translation(
                SO3.from_x_radians(np.pi * 0.5), np.array([0.5, 0, 0])
            ),
        )
        assert_transforms_close(
            start.interpolate(end, alpha=0.75),
            SE3.from_rotation_and_translation(
                SO3.from_x_radians(np.pi * 0.75), np.array([0.75, 0, 0])
            ),
        )
        assert_transforms_close(start.interpolate(end, alpha=0.0), start)
        assert_transforms_close(start.interpolate(end, alpha=1.0), end)

        expected_error_message = "Expected alpha within [0.0, 1.0]"
        with self.assertRaises(ValueError) as cm:
            start.interpolate(end, alpha=-1.0)
        self.assertIn(expected_error_message, str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            start.interpolate(end, alpha=2.0)
        self.assertIn(expected_error_message, str(cm.exception))

    def test_se3_clamp(self):
        T = SE3.from_rotation_and_translation(
            rotation=SO3.from_rpy_radians(roll=np.pi, pitch=0, yaw=-np.pi),
            translation=np.array([0, 1, 2]),
        )
        original_rpy = T.rotation().as_rpy_radians()

        # Clamping with the default limits (+- infinity) means the SE3 should remain unchanged.
        self.assertEqual(T, T.clamp())

        # Test clamping the x translation.
        clamped_T = T.clamp(x_translation=(1, 2))
        np.testing.assert_allclose(clamped_T.translation(), np.array([1, 1, 2]))
        self.assertEqual(clamped_T.rotation(), T.rotation())

        # Test clamping the y translation.
        clamped_T = T.clamp(y_translation=(-1, 0))
        np.testing.assert_allclose(clamped_T.translation(), np.array([0, 0, 2]))
        self.assertEqual(clamped_T.rotation(), T.rotation())

        # Test clamping the z translation.
        clamped_T = T.clamp(z_translation=(5, 10))
        np.testing.assert_allclose(clamped_T.translation(), np.array([0, 1, 5]))
        self.assertEqual(clamped_T.rotation(), T.rotation())

        # Test clamping the roll.
        clamped_T = T.clamp(roll_radians=(0, 1))
        clamped_rpy = clamped_T.rotation().as_rpy_radians()
        np.testing.assert_equal(clamped_T.translation(), T.translation())
        self.assertAlmostEqual(clamped_rpy.roll, 1)
        self.assertAlmostEqual(clamped_rpy.pitch, original_rpy.pitch)
        self.assertAlmostEqual(clamped_rpy.yaw, original_rpy.yaw)

        # Test clamping the pitch.
        clamped_T = T.clamp(pitch_radians=(1, 2))
        clamped_rpy = clamped_T.rotation().as_rpy_radians()
        np.testing.assert_equal(clamped_T.translation(), T.translation())
        self.assertAlmostEqual(clamped_rpy.roll, original_rpy.roll)
        self.assertAlmostEqual(clamped_rpy.pitch, 1)
        self.assertAlmostEqual(clamped_rpy.yaw, original_rpy.yaw)

        # Test clamping the yaw.
        clamped_T = T.clamp(yaw_radians=(np.pi, 2 * np.pi))
        clamped_rpy = clamped_T.rotation().as_rpy_radians()
        np.testing.assert_equal(clamped_T.translation(), T.translation())
        self.assertAlmostEqual(clamped_rpy.roll, original_rpy.roll)
        self.assertAlmostEqual(clamped_rpy.pitch, original_rpy.pitch)
        self.assertAlmostEqual(clamped_rpy.yaw, np.pi)


class TestHashAndSetMembership(absltest.TestCase):
    """Test that SO3 and SE3 objects can be hashed and used in sets."""

    def test_so3_hash_and_set_membership(self):
        a = SO3.from_rpy_radians(0.1, -0.2, 0.3)
        b = SO3(wxyz=a.wxyz.copy())
        c = SO3.from_rpy_radians(0.1, -0.2, 0.31)
        assert a == b
        assert hash(a) == hash(b)
        s = {a, b, c}
        assert a in s and b in s and c in s
        assert len(s) == 2

    def test_se3_hash_and_set_membership(self):
        R = SO3.from_rpy_radians(0.05, 0.02, -0.01)
        t = np.array([0.3, -0.1, 0.2], dtype=np.float64)
        a = SE3.from_rotation_and_translation(R, t)
        b = SE3.from_rotation_and_translation(SO3(wxyz=R.wxyz.copy()), t.copy())
        c = SE3.from_rotation_and_translation(R, t + np.array([1e-3, 0.0, 0.0]))
        assert a == b
        assert hash(a) == hash(b)
        s = {a, b, c}
        assert len(s) == 2


class TestSE3_getQ(absltest.TestCase):
    """Covers both small-angle and general branches in mink.lie.se3._getQ."""

    def test__getQ_small_angle_zero(self):
        # theta == 0 (small-angle branch); with v=0 too, Q should be exactly zero.
        c_small = np.zeros(6, dtype=np.float64)
        Q = _getQ(c_small)
        np.testing.assert_allclose(Q, np.zeros((3, 3), dtype=np.float64))

    def test__getQ_general_branch_nontrivial(self):
        # Non-zero rotation (general branch).
        # Use non-zero v to avoid the trivial zero result from the small-angle test.
        c = np.zeros(6, dtype=np.float64)
        c[:3] = np.array([0.3, -0.2, 0.1], dtype=np.float64)  # v
        c[3:] = np.array([0.4, 0.0, 0.0], dtype=np.float64)  # omega (theta ≈ 0.4 > 0)

        Q = _getQ(c)
        self.assertEqual(Q.shape, (3, 3))
        self.assertTrue(np.isfinite(Q).all())

        # Sanity: changing v (with same omega) should change Q.
        c_scaled = c.copy()
        c_scaled[:3] *= 2.0
        Q_scaled = _getQ(c_scaled)
        self.assertGreater(np.linalg.norm(Q - Q_scaled), 1e-9)


if __name__ == "__main__":
    absltest.main()
