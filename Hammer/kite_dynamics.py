"""
kite_dynamics.py — 6-DOF Rigid-Body Equations of Motion & Radiomaster Mixer
===========================================================================
State Vector (13 elements):
  [0:3]   x, y, z        : World position (m) — +X Forward, +Y Right, +Z UP (Altitude)
  [3:6]   u, v, w        : Body velocities (m/s) — +u Fwd, +v Right, +w Down
  [6:10]  q0, q1, q2, q3 : Unit quaternion (Body -> World NED orientation)
  [10:13] p, q, r        : Body angular rates (rad/s) — Roll, Pitch, Yaw
"""
import math
from kite_aero import AircraftGeometry, FlightState, ControlInputs, ModularAeroModel


class RadiomasterMixer:
    """Converts normalized Radiomaster RC sticks (-1..+1) or UI sliders into elevon angles."""
    def __init__(self, max_pitch_deg: float = 20.0, max_roll_deg: float = 18.0):
        self.max_pitch_deg = max_pitch_deg
        self.max_roll_deg = max_roll_deg

    def mix(self, throttle_pct: float, pitch_deg: float, roll_deg: float = 0.0) -> ControlInputs:
        # Delta-wing elevon mixing:
        # +pitch_deg = trailing edge down (+de_l, +de_r)
        # +roll_deg (roll right) = left elevon down (+de_l), right elevon up (-de_r)
        de_l = max(-25.0, min(25.0, pitch_deg + roll_deg))
        de_r = max(-25.0, min(25.0, pitch_deg - roll_deg))
        return ControlInputs(
            throttle=max(0.0, min(1.0, throttle_pct / 100.0)),
            elevon_l_deg=de_l,
            elevon_r_deg=de_r
        )


def euler_to_quat(phi: float, theta: float, psi: float) -> list[float]:
    cp, sp = math.cos(phi * 0.5), math.sin(phi * 0.5)
    ct, st = math.cos(theta * 0.5), math.sin(theta * 0.5)
    cy, sy = math.cos(psi * 0.5), math.sin(psi * 0.5)
    return [
        cp * ct * cy + sp * st * sy,
        sp * ct * cy - cp * st * sy,
        cp * st * cy + sp * ct * sy,
        cp * ct * sy - sp * st * cy,
    ]


def quat_to_euler(q0: float, q1: float, q2: float, q3: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
    phi = math.atan2(sinr_cosp, cosr_cosp)

    sinp = max(-1.0, min(1.0, 2.0 * (q0 * q2 - q3 * q1)))
    theta = math.asin(sinp)

    siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
    psi = math.atan2(siny_cosp, cosy_cosp)
    return phi, theta, psi


class SixDOFSimulator:
    def __init__(self, aero_model: ModularAeroModel = None):
        self.aero = aero_model or ModularAeroModel()
        self.geom = self.aero.geom
        self.mixer = RadiomasterMixer()

    def state_to_flight_state(self, s: list[float]) -> FlightState:
        x, y, z, u, v, w, q0, q1, q2, q3, p, q, r = s
        phi, theta, psi = quat_to_euler(q0, q1, q2, q3)
        return FlightState(x=x, y=y, z=z, u=u, v=v, w=w, p=p, q=q, r=r,
                           phi=phi, theta=theta, psi=psi)

    def derivs(self, s: list[float], ctrl: ControlInputs) -> list[float]:
        x, y, z, u, v, w, q0, q1, q2, q3, p, q, r = s
        g = self.geom
        fs = self.state_to_flight_state(s)
        wrench, _ = self.aero.evaluate(fs, ctrl)

        # Gravity components in Body Frame (+Z down)
        gx = -g.g * 2.0 * (q1 * q3 - q0 * q2)
        gy =  g.g * 2.0 * (q2 * q3 + q0 * q1)
        gz =  g.g * (q0*q0 - q1*q1 - q2*q2 + q3*q3)

        # 1. Translational Accelerations (Body Frame)
        du = (wrench.fx / g.mass) + gx - (q * w - r * v)
        dv = (wrench.fy / g.mass) + gy - (r * u - p * w)
        dw = (wrench.fz / g.mass) + gz - (p * v - q * u)

        # 2. World Position Derivatives (Body -> World with +Z = Altitude UP)
        r11 = q0*q0 + q1*q1 - q2*q2 - q3*q3
        r12 = 2.0 * (q1*q2 - q0*q3)
        r13 = 2.0 * (q1*q3 + q0*q2)
        r21 = 2.0 * (q1*q2 + q0*q3)
        r22 = q0*q0 - q1*q1 + q2*q2 - q3*q3
        r23 = 2.0 * (q2*q3 - q0*q1)
        r31 = 2.0 * (q1*q3 - q0*q2)
        r32 = 2.0 * (q2*q3 + q0*q1)
        r33 = q0*q0 - q1*q1 - q2*q2 + q3*q3

        dx = r11 * u + r12 * v + r13 * w
        dy = r21 * u + r22 * v + r23 * w
        vnd = r31 * u + r32 * v + r33 * w
        dz = -vnd  # Convert NED Down velocity to Altitude UP velocity

        # 3. Quaternion Kinematics
        dq0 = 0.5 * (-p * q1 - q * q2 - r * q3)
        dq1 = 0.5 * ( p * q0 + r * q2 - q * q3)
        dq2 = 0.5 * ( q * q0 - r * q1 + p * q3)
        dq3 = 0.5 * ( r * q0 + q * q1 - p * q2)

        # 4. Rotational Accelerations (Euler's Equations with Ixz cross-coupling)
        Ixx, Iyy, Izz, Ixz = g.Ixx, g.Iyy, g.Izz, g.Ixz
        gamma = Ixx * Izz - Ixz * Ixz

        # Gyroscopic inertia coupling
        L_tot = wrench.l - (Izz - Iyy) * q * r - Ixz * p * q
        M_tot = wrench.m - (Ixx - Izz) * p * r - Ixz * (p*p - r*r)
        N_tot = wrench.n - (Iyy - Ixx) * p * q + Ixz * q * r

        dp = (Izz * L_tot + Ixz * N_tot) / gamma
        dq = M_tot / Iyy
        dr = (Ixz * L_tot + Ixx * N_tot) / gamma

        return [dx, dy, dz, du, dv, dw, dq0, dq1, dq2, dq3, dp, dq, dr]

    def step_rk4(self, s: list[float], ctrl: ControlInputs, dt: float) -> list[float]:
        k1 = self.derivs(s, ctrl)
        s2 = [si + 0.5 * dt * ki for si, ki in zip(s, k1)]
        k2 = self.derivs(s2, ctrl)
        s3 = [si + 0.5 * dt * ki for si, ki in zip(s, k2)]
        k3 = self.derivs(s3, ctrl)
        s4 = [si + dt * ki for si, ki in zip(s, k3)]
        k4 = self.derivs(s4, ctrl)

        s_next = [si + (dt / 6.0) * (a + 2*b + 2*c + d)
                  for si, a, b, c, d in zip(s, k1, k2, k3, k4)]

        # Normalize quaternion to prevent numerical drift
        qnorm = math.sqrt(sum(q*q for q in s_next[6:10])) or 1.0
        for i in range(6, 10):
            s_next[i] /= qnorm
        return s_next

    def simulate_trajectory(self, launch_v: float, rail_deg: float, throttle_pct: float,
                            elevon_deg: float, roll_deg: float, sim_time: float,
                            launch_h: float = 1.0, fps: int = 24, dt: float = 0.004) -> list[dict]:
        ang = math.radians(rail_deg)
        q_init = euler_to_quat(0.0, ang, 0.0)
        # Catapult launches along pitch angle theta = ang (so body u = V, w = 0)
        state = [0.0, 0.0, launch_h, launch_v, 0.0, 0.0, *q_init, 0.0, 0.0, 0.0]
        ctrl = self.mixer.mix(throttle_pct, elevon_deg, roll_deg)

        frame_dt = 1.0 / fps
        spf = max(1, int(frame_dt / dt))
        adt = frame_dt / spf
        n_frames = int(sim_time * fps)

        frames = []
        t = 0.0
        crashed = False

        for fi in range(n_frames + 1):
            fs = self.state_to_flight_state(state)
            wrench, ctx = self.aero.evaluate(fs, ctrl)

            frames.append(dict(
                frame=fi + 1, t=t,
                x=fs.x, y=fs.y, z=fs.z,
                V=fs.V, Vx=fs.u, Vz=-fs.w,
                phi=fs.phi, theta=fs.theta, psi=fs.psi,
                p=fs.p, q=fs.q, r=fs.r,
                alpha=fs.alpha, beta=fs.beta,
                alpha_deg=math.degrees(fs.alpha),
                beta_deg=math.degrees(fs.beta),
                CL=ctx["CL"], CD=ctx["CD"], Cm=ctx["Cm"],
                CL_pot=ctx.get("CL_pot", 0.0), CL_vort=ctx.get("CL_vort", 0.0),
                stalled=ctx.get("stalled", False),
                crashed=crashed
            ))
            if crashed:
                break

            for _ in range(spf):
                state = self.step_rk4(state, ctrl, adt)
            t += frame_dt

            if state[2] <= 0.0 and t > 0.15:
                state[2] = 0.0
                crashed = True

        return frames