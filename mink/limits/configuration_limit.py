"""Joint position limit."""

from typing import List

import mujoco
import numpy as np

from ..configuration import Configuration
from ..constants import dof_width, qpos_width
from ..exceptions import LimitDefinitionError
from .limit import Constraint, Limit


class ConfigurationLimit(Limit):
    """Inequality constraint on joint positions in a robot model.

    Floating base joints are ignored.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        gain: float = 0.95,
        min_distance_from_limits: float = 0.0,
    ):
        """Initialize configuration limits.

        Args:
            model: MuJoCo model.
            gain: Gain factor in (0, 1] that determines how fast each joint is
                allowed to move towards the joint limits at each timestep. Values lower
                ttan 1 are safer but may make the joints move slowly.
            min_distance_from_limits: Offset in meters (slide joints) or radians
                (hinge joints) to be added to the limits. Positive values decrease the
                range of motion, negative values increase it (i.e. negative values
                allow penetration).
        """
        if not 0.0 < gain <= 1.0:
            raise LimitDefinitionError(
                f"{self.__class__.__name__} gain must be in the range (0, 1]"
            )

        index_list: List[int] = []  # DoF indices that are limited.
        lower = np.full(model.nq, -mujoco.mjMAXVAL)
        upper = np.full(model.nq, mujoco.mjMAXVAL)
        for jnt in range(model.njnt):
            jnt_type = model.jnt_type[jnt]
            qpos_dim = qpos_width(jnt_type)
            jnt_range = model.jnt_range[jnt]
            padr = model.jnt_qposadr[jnt]
            # Skip free joints and joints without limits.
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE or not model.jnt_limited[jnt]:
                continue
            lower[padr : padr + qpos_dim] = jnt_range[0] + min_distance_from_limits
            upper[padr : padr + qpos_dim] = jnt_range[1] - min_distance_from_limits
            jnt_dim = dof_width(jnt_type)
            jnt_id = model.jnt_dofadr[jnt]
            index_list.extend(range(jnt_id, jnt_id + jnt_dim))

        self.indices = np.array(index_list)
        self.indices.setflags(write=False)

        dim = len(self.indices)
        self.projection_matrix = np.eye(model.nv)[self.indices] if dim > 0 else None

        self.lower = lower
        self.upper = upper
        self.model = model
        self.gain = gain

    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        r"""Compute the configuration-dependent joint position limits.

        The limits are defined as:

        .. math::

            {q \ominus q_{min}} \leq \Delta q \leq {q_{max} \ominus q}

        where :math:`q \in {\cal C}` is the robot's configuration and
        :math:`\Delta q \in T_q({\cal C})` is the displacement in the tangent
        space at :math:`q`. See the :ref:`derivations` section for more information.

        Args:
            configuration: Robot configuration :math:`q`.
            dt: Integration timestep in [s].

        Returns:
            Pair :math:`(G, h)` representing the inequality constraint as
            :math:`G \Delta q \leq h`, or ``None`` if there is no limit.
        """
        if self.projection_matrix is None:
            return Constraint()

        # NOTE: with dt=1.0, mj_differentiatePos computes the velocity that would
        # take us from qpos1 to qpos2 in one second.
        # Upper.
        vel_to_upper_limit = np.empty((self.model.nv,))
        mujoco.mj_differentiatePos(
            m=self.model,
            qvel=vel_to_upper_limit,
            dt=1.0,
            qpos1=configuration.q,
            qpos2=self.upper,
        )

        # Lower.
        vel_from_lower_limit = np.empty((self.model.nv,))
        mujoco.mj_differentiatePos(
            m=self.model,
            qvel=vel_from_lower_limit,
            dt=1.0,
            # NOTE: mj_differentiatePos does `qpos2 - qpos1` so notice the order
            # swap here compared to above.
            qpos1=self.lower,
            qpos2=configuration.q,
        )

        # The QP solves for joint displacements, so we need to scale the joint
        # velocity limits by the timestep dt.
        # The gain is a safety factor to prevent the joint from moving too close
        # to the limit.
        max_displacement = self.gain * vel_to_upper_limit[self.indices] * dt
        min_displacement = self.gain * vel_from_lower_limit[self.indices] * dt

        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        h = np.hstack([max_displacement, min_displacement])
        return Constraint(G=G, h=h)
