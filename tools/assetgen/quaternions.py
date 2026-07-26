"""Quaternion helpers for authoring glTF node rotations.

glTF stores rotations as `[x, y, z, w]`, and every function here uses that order — the
component order is the single easiest thing to get wrong when hand-authoring a scene.

Extracted verbatim from `assets/physics_demos/generate.py`.
"""

import math

from .geometry import normalise, vec_cross, vec_sub


def quat_axis_angle(axis, angle):
    """Quaternion [x,y,z,w] for a rotation of `angle` radians about a (unit-ish) axis."""
    m = math.sqrt(sum(c * c for c in axis)) or 1.0
    ax, ay, az = (axis[0] / m, axis[1] / m, axis[2] / m)
    s = math.sin(angle / 2.0)
    return (ax * s, ay * s, az * s, math.cos(angle / 2.0))


def rotate_by_quat(q, v):
    """Rotate vec3 v by quaternion q = [x,y,z,w]."""
    qx, qy, qz, qw = q
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (qy * v[2] - qz * v[1])
    ty = 2.0 * (qz * v[0] - qx * v[2])
    tz = 2.0 * (qx * v[1] - qy * v[0])
    # v + qw * t + cross(q.xyz, t)
    return (
        v[0] + qw * tx + (qy * tz - qz * ty),
        v[1] + qw * ty + (qz * tx - qx * tz),
        v[2] + qw * tz + (qx * ty - qy * tx),
    )


def quat_conjugate(q):
    """Conjugate/inverse of a unit quaternion q = [x,y,z,w]."""
    return (-q[0], -q[1], -q[2], q[3])


def quat_multiply(a, b):
    """Hamilton product for glTF quaternions [x,y,z,w]."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_from_to(a, b):
    """Shortest quaternion rotating vector `a` onto vector `b`."""
    av = normalise(a)
    bv = normalise(b)
    dot = max(-1.0, min(1.0, sum(av[i] * bv[i] for i in range(3))))
    if dot > 0.999999:
        return (0.0, 0.0, 0.0, 1.0)
    if dot < -0.999999:
        axis = vec_cross(av, (1.0, 0.0, 0.0))
        if sum(c * c for c in axis) < 1.0e-8:
            axis = vec_cross(av, (0.0, 0.0, 1.0))
        return quat_axis_angle(axis, math.pi)
    axis = vec_cross(av, bv)
    s = math.sqrt((1.0 + dot) * 2.0)
    inv_s = 1.0 / s
    return (axis[0] * inv_s, axis[1] * inv_s, axis[2] * inv_s, 0.5 * s)


def look_at_quat(eye, target, up):
    """Quaternion [x,y,z,w] orienting a glTF camera (looks down -Z) from eye to target."""
    z = normalise(vec_sub(eye, target))  # camera +Z points away from the target
    x = normalise(vec_cross(up, z))
    y = vec_cross(z, x)
    # Rotation matrix columns (x, y, z) -> quaternion.
    m00, m01, m02 = x[0], y[0], z[0]
    m10, m11, m12 = x[1], y[1], z[1]
    m20, m21, m22 = x[2], y[2], z[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return (qx, qy, qz, w)
