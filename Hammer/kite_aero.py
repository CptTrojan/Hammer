"""
kite_aero.py — Modular 6-DOF Aerodynamics & Propulsion Engine for KITE UAV
==========================================================================
Body Frame Convention (Standard Flight Dynamics):
  +X : Forward (out the nose)
  +Y : Starboard (out the right wing)
  +Z : Down (through the belly)
Forces:  [Fx, Fy, Fz] in Newtons
Moments: [L (roll), M (pitch), N (yaw)] in N*m about the CG
"""
import math
from dataclasses import dataclass

@dataclass
class AeroWrench:
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0
    l: float = 0.0   # Roll moment (+ right wing down)
    m: float = 0.0   # Pitch moment (+ nose up)
    n: float = 0.0   # Yaw moment (+ nose right)

    def __iadd__(self, other: "AeroWrench"):
        self.fx += other.fx
        self.fy += other.fy
        self.fz += other.fz
        self.l  += other.l
        self.m  += other.m
        self.n  += other.n
        return self


@dataclass
class AircraftGeometry:
    """Exact geometry extracted from kite_canard_delta.py (in SI units)."""
    mass: float = 1.500
    g: float = 9.81
    rho: float = 1.225

    # Inertia Tensor (kg*m^2) estimated for 900mm 3D-printed canard-delta
    Ixx: float = 0.014
    Iyy: float = 0.022
    Izz: float = 0.034
    Ixz: float = 0.0012

    # Main Wing (Cranked/Swept Delta)
    span: float = 0.800
    c_root: float = 0.390
    c_tip: float = 0.150
    le_root: float = 0.300
    le_tip: float = 0.550
    wing_z: float = 0.000       # 6 mm below centerline (+Z is down)

    # Canard
    span_can: float = 0.386
    c_can: float = 0.100
    x_can_le: float = 0.105

    # Elevons
    elev_chord: float = 0.075
    elev_y_in: float = 0.140
    elev_y_out: float = 0.380

    # Tip Fins (Drooped)
    fin_area_each: float = 0.0015
    fin_x_ac: float = 0.710
    fin_z_ac: float = 0.020     # Drooped below wing (+Z down)

    # CG & Propulsion (A2212 1000KV + 1045 Pusher Prop)
    x_cg: float = 0.430
    motor_x: float = 0.716      # Prop station aft of nose
    motor_z: float = -0.012     # 12 mm ABOVE centerline (-Z is up in body frame)
    prop_d: float = 0.254       # 10 inch prop (0.254 m)
    prop_pitch: float = 0.1143  # 4.5 inch pitch
    I_prop: float = 4.5e-5      # Rotational inertia of 1045 prop + bell (kg*m^2)
    max_rpm: float = 9200.0     # 3S LiPo loaded RPM

    def __post_init__(self):
        self.semi = self.span / 2.0
        self.s_wing = 0.5 * (self.c_root + self.c_tip) * self.span
        self.s_ref = self.s_wing
        self.taper = self.c_tip / self.c_root
        self.ar = (self.span ** 2) / self.s_wing
        self.sweep_le = math.atan2(self.le_tip - self.le_root, self.semi)

        t = self.taper
        self.mac = (2.0 / 3.0) * self.c_root * (1 + t + t**2) / (1 + t)
        self.y_mac = (self.semi / 3.0) * (1 + 2 * t) / (1 + t)
        self.x_mac_le = self.le_root + (self.le_tip - self.le_root) * (self.y_mac / self.semi)
        self.x_ac_wing = self.x_mac_le + 0.25 * self.mac

        self.s_can = self.span_can * self.c_can
        self.ar_can = (self.span_can ** 2) / self.s_can
        self.x_ac_can = self.x_can_le + 0.25 * self.c_can


@dataclass
class FlightState:
    """Instantaneous 6-DOF state passed to aerodynamic modules."""
    x: float = 0.0          # World North/Forward (m)
    y: float = 0.0          # World East/Right (m)
    z: float = 1.0          # World Altitude UP (m)
    u: float = 12.0         # Body forward velocity (m/s)
    v: float = 0.0          # Body right velocity (m/s)
    w: float = 0.0          # Body down velocity (m/s)
    p: float = 0.0          # Roll rate (rad/s)
    q: float = 0.0          # Pitch rate (rad/s)
    r: float = 0.0          # Yaw rate (rad/s)
    phi: float = 0.0        # Roll angle (rad)
    theta: float = 0.0      # Pitch angle (rad)
    psi: float = 0.0        # Yaw angle (rad)

    @property
    def V(self) -> float:
        return max(0.05, math.sqrt(self.u**2 + self.v**2 + self.w**2))

    @property
    def alpha(self) -> float:
        return max(min(math.atan2(self.w, max(0.05, abs(self.u))), math.radians(65)), math.radians(-65))

    @property
    def beta(self) -> float:
        return max(min(math.asin(max(-1.0, min(1.0, self.v / self.V))), math.radians(45)), math.radians(-45))

    @property
    def alpha_deg(self) -> float:
        return math.degrees(self.alpha)

    @property
    def beta_deg(self) -> float:
        return math.degrees(self.beta)


@dataclass
class ControlInputs:
    """Pilot / Radiomaster inputs."""
    throttle: float = 1.0       # 0.0 to 1.0
    pitch_cmd: float = 0.0      # -1.0 to +1.0 (stick) OR direct degrees via mixer
    roll_cmd: float = 0.0       # -1.0 to +1.0
    yaw_cmd: float = 0.0        # -1.0 to +1.0
    elevon_l_deg: float = 0.0   # Left elevon (+ trailing edge down)
    elevon_r_deg: float = 0.0   # Right elevon (+ trailing edge down)


# ===================================================================
#  MODULAR PHENOMENA CLASSES
# ===================================================================

class AeroModule:
    enabled: bool = True
    def compute(self, state: FlightState, ctrl: ControlInputs, g: AircraftGeometry, ctx: dict) -> AeroWrench:
        return AeroWrench()


class PolhamusWingModule(AeroModule):
    """Main delta wing potential + vortex lift, asymmetric vortex breakdown, & damping."""
    def __init__(self, g: AircraftGeometry):
        tanL = math.tan(g.sweep_le)
        self.Kp = (2 * math.pi * g.ar) / (2 + math.sqrt(4 + g.ar**2 * (1 + tanL**2)))
        self.Kv = 0.65 * self.Kp  # Rounded 3D-printed LE vortex factor
        self.alpha_vb = math.radians(28.0 + 3.0 * (g.ar - 2.0))

    def compute(self, s: FlightState, ctrl: ControlInputs, g: AircraftGeometry, ctx: dict) -> AeroWrench:
        a = s.alpha
        b = s.beta
        V = s.V
        qbar = 0.5 * g.rho * V**2

        # Apply canard downwash if canard module ran first
        a_eff = a - ctx.get("canard_downwash_rad", 0.0)
        sa, ca = math.sin(a_eff), math.cos(a_eff)

        # Polhamus Potential + Vortex Lift
        cl_pot = self.Kp * sa * ca
        cl_vort = self.Kv * ca * (sa ** 2) * math.copysign(1.0, a_eff)

        # Sideslip lowers vortex breakdown angle on windward wing
        vb_eff = self.alpha_vb - 0.6 * abs(b)
        stalled = abs(a_eff) > vb_eff
        if stalled:
            frac = min(1.0, (abs(a_eff) - vb_eff) / math.radians(15.0))
            cl_vort *= (1.0 - 0.75 * frac)

        cl_wing = cl_pot + cl_vort
        ctx["CL_wing"] = cl_wing
        ctx["CL_pot"] = cl_pot
        ctx["CL_vort"] = cl_vort
        ctx["stalled"] = stalled

        # Parasitic + Induced + Vortex + Post-Stall Separation Drag
        cd0 = 0.018
        e_oswald = max(0.35, 0.68 - 0.15 * (abs(a_eff) / math.radians(30.0)))
        ge_factor = ctx.get("ground_effect_factor", 1.0)
        cdi = (cl_wing ** 2) / (math.pi * g.ar * e_oswald) * ge_factor
        cd_vort = 0.02 * max(0.0, abs(sa) - math.sin(math.radians(8.0)))**2
        cd_sep = 1.2 * math.sin(abs(a_eff) - vb_eff)**2 if stalled else 0.0
        cd_wing = cd0 + cdi + cd_vort + cd_sep
        ctx["CD_wing"] = cd_wing

        # Convert Wind/Stability Lift & Drag to Body Fx, Fz
        L_force = qbar * g.s_ref * cl_wing
        D_force = qbar * g.s_ref * cd_wing
        fx = L_force * math.sin(a) - D_force * math.cos(a)
        fz = -L_force * math.cos(a) - D_force * math.sin(a)

        # Wing pitching moment about CG + Pitch Damping (Cm_q)
        arm_x = g.x_cg - g.x_ac_wing  # Positive if CG is aft of AC (unstable nose-up)
        m_wing = fz * (-arm_x)        # Since fz is negative for positive lift, fz * (x_ac - x_cg)
        m_wing = -L_force * (g.x_ac_wing - g.x_cg)
        if stalled:
            m_wing -= qbar * g.s_ref * g.mac * 0.15 * math.copysign(abs(a_eff) - vb_eff, a_eff)

        cm_q = -8.5 * (s.q * g.mac / (2.0 * V))
        m_damp = qbar * g.s_ref * g.mac * cm_q

        # Lateral-Directional: Swept Delta Dihedral Effect (Cl_beta) & Roll Damping (Cl_p)
        # Swept delta wings have strong effective dihedral proportional to CL
        cl_beta = -(0.12 + 0.22 * cl_wing * math.sin(g.sweep_le)) * b
        cl_p = -0.32 * (s.p * g.span / (2.0 * V))
        l_roll = qbar * g.s_ref * g.span * (cl_beta + cl_p)

        return AeroWrench(fx=fx, fz=fz, l=l_roll, m=m_wing + m_damp)


class CanardModule(AeroModule):
    """Forward canard lift, pitch moment, and downwash interference on main wing."""
    def __init__(self, g: AircraftGeometry):
        self.cl_a_can = (2 * math.pi / (1.0 + 2.0 / g.ar_can)) * (g.s_can / g.s_ref)
        self.alpha_stall_can = math.radians(16.0)  # Canard stalls before delta wing (safe nose-drop!)

    def compute(self, s: FlightState, ctrl: ControlInputs, g: AircraftGeometry, ctx: dict) -> AeroWrench:
        a = s.alpha
        V = s.V
        qbar = 0.5 * g.rho * V**2

        # Local pitch rate induces extra alpha at forward canard
        arm_can = g.x_cg - g.x_ac_can  # Positive (canard is ahead of CG)
        a_can = a - (s.q * arm_can / V)

        if abs(a_can) <= self.alpha_stall_can:
            cl_can = self.cl_a_can * a_can
        else:
            # Canard stall drops nose automatically (passive stall recovery)
            peak = self.cl_a_can * math.copysign(self.alpha_stall_can, a_can)
            excess = abs(a_can) - self.alpha_stall_can
            cl_can = peak * max(0.2, 1.0 - excess / math.radians(12.0))

        ctx["CL_can"] = cl_can
        # Store downwash angle for the wing module
        ctx["canard_downwash_rad"] = 0.14 * cl_can

        cd_can = 0.003 + (cl_can ** 2) / (math.pi * g.ar_can * 0.75) * (g.s_ref / g.s_can)
        ctx["CD_can"] = cd_can

        L_can = qbar * g.s_ref * cl_can
        D_can = qbar * g.s_ref * cd_can

        fx = L_can * math.sin(a) - D_can * math.cos(a)
        fz = -L_can * math.cos(a) - D_can * math.sin(a)
        m_can = L_can * arm_can  # Canard ahead of CG produces +M (nose-up)

        return AeroWrench(fx=fx, fz=fz, m=m_can)


class ElevonControlModule(AeroModule):
    """Split left/right elevon deflections producing Pitch, Roll, and Adverse Yaw."""
    def __init__(self, g: AircraftGeometry):
        cr = g.elev_chord / g.c_root
        span_ratio = 2.0 * (g.elev_y_out - g.elev_y_in) / g.span
        self.cl_de = 2.0 * math.pi * cr * span_ratio * math.sqrt(max(0.0, 1.0 - (1.0 - cr)**2))
        self.x_hinge = 0.700 - 0.5 * g.elev_chord
        self.y_mid = 0.5 * (g.elev_y_in + g.elev_y_out)

    def compute(self, s: FlightState, ctrl: ControlInputs, g: AircraftGeometry, ctx: dict) -> AeroWrench:
        V = s.V
        qbar = 0.5 * g.rho * V**2

        de_l = math.radians(ctrl.elevon_l_deg)
        de_r = math.radians(ctrl.elevon_r_deg)

        # Symmetric = Elevator (+TE down = +Lift, -Pitch Moment)
        de_sym = 0.5 * (de_l + de_r)
        # Differential = Aileron (+left down / -right up = +Right Roll Moment)
        de_diff = 0.5 * (de_l - de_r)

        cl_elev = self.cl_de * de_sym
        cd_elev = 0.015 * (de_l**2 + de_r**2)
        ctx["CL_elev"] = cl_elev
        ctx["CD_elev"] = cd_elev

        L_elev = qbar * g.s_ref * cl_elev
        D_elev = qbar * g.s_ref * cd_elev

        fx = L_elev * math.sin(s.alpha) - D_elev * math.cos(s.alpha)
        fz = -L_elev * math.cos(s.alpha) - D_elev * math.sin(s.alpha)

        # Pitch moment from symmetric deflection
        arm_x = self.x_hinge - g.x_cg
        m_pitch = -L_elev * arm_x

        # Roll moment from differential deflection
        L_diff = qbar * g.s_ref * self.cl_de * de_diff
        l_roll = L_diff * self.y_mid

        # Differential drag adverse yaw
        D_diff = qbar * g.s_ref * 0.018 * (de_l**2 - de_r**2)
        n_yaw = -D_diff * self.y_mid

        return AeroWrench(fx=fx, fz=fz, l=l_roll, m=m_pitch, n=n_yaw)


class PropulsionAndTorqueModule(AeroModule):
    """Pusher propeller thrust, motor reaction torque, high-pylon pitch moment, & gyro precession."""
    def __init__(self, g: AircraftGeometry):
        self.t_static_max = 1.2 * g.mass * g.g  # 17.66 N static max

    def compute(self, s: FlightState, ctrl: ControlInputs, g: AircraftGeometry, ctx: dict) -> AeroWrench:
        thr = max(0.0, min(1.0, ctrl.throttle))
        if thr < 0.001:
            ctx["T"] = 0.0
            return AeroWrench()

        # Propeller angular velocity (rad/s)
        rpm = thr * g.max_rpm
        n_rps = rpm / 60.0
        omega_prop = rpm * 2.0 * math.pi / 60.0

        # Advance Ratio J = V / (n * D) -> thrust decays as airspeed approaches pitch speed
        J = s.V / max(1.0, n_rps * g.prop_d)
        j_factor = max(0.0, 1.0 - 0.55 * (J ** 1.4))
        thrust = self.t_static_max * (thr ** 2) * j_factor
        ctx["T"] = thrust

        # 1. High-mounted pusher pylon pitching moment:
        # Motor sits above CG (motor_z = -0.012 m in body Z-down frame).
        # Forward thrust above CG creates a NOSE-DOWN (-M) pitching moment: M = z_motor * Fx
        m_pylon = g.motor_z * thrust

        # 2. Motor Reaction Torque (Newton's 3rd Law):
        # Prop Torque Q = C_Q * rho * n^2 * D^5 (or related to thrust via pitch/diameter ratio)
        q_motor = thrust * (g.prop_d / 2.0) * 0.14
        l_torque = -q_motor  # Counter-torque rolls the airframe opposite to prop spin

        # 3. Gyroscopic Precession of Spinning Propeller + Outrunner Bell:
        # M_gyro = -omega_vec x (I_prop * rate_vec)
        m_gyro = g.I_prop * omega_prop * s.r
        n_gyro = -g.I_prop * omega_prop * s.q

        return AeroWrench(
            fx=thrust,
            l=l_torque,
            m=m_pylon + m_gyro,
            n=n_gyro
        )


class TipFinAndFuselageModule(AeroModule):
    """Drooped wingtip fins + fuselage sideforce (CY_beta), weathercock stability (Cn_beta), & yaw damping."""
    def compute(self, s: FlightState, ctrl: ControlInputs, g: AircraftGeometry, ctx: dict) -> AeroWrench:
        V = s.V
        qbar = 0.5 * g.rho * V**2
        b = s.beta

        # Sideforce derivative (negative beta = wind from right -> pushes left)
        cy_b = -0.28 * b
        fy = qbar * g.s_ref * cy_b

        # Directional weathercock stability (Cn_beta > 0) + Yaw rate damping (Cn_r < 0)
        cn_b = 0.065 * b
        cn_r = -0.095 * (s.r * g.span / (2.0 * V))
        n_yaw = qbar * g.s_ref * g.span * (cn_b + cn_r)

        # Drooped fins sit below CG (+Z), so sideforce creates a small roll couple
        l_fin = -fy * g.fin_z_ac

        return AeroWrench(fy=fy, l=l_fin, n=n_yaw)


class GroundEffectModule(AeroModule):
    """Wieselsberger / McCormick ground effect reducing induced drag near ground."""
    def compute(self, s: FlightState, ctrl: ControlInputs, g: AircraftGeometry, ctx: dict) -> AeroWrench:
        h = max(0.05, s.z)
        hb_ratio = h / g.span
        if hb_ratio < 1.2:
            num = (16.0 * hb_ratio) ** 2
            ctx["ground_effect_factor"] = num / (1.0 + num)
        else:
            ctx["ground_effect_factor"] = 1.0
        return AeroWrench()


# ===================================================================
#  MASTER PIPELINE AGGREGATOR
# ===================================================================

class ModularAeroModel:
    def __init__(self, geom: AircraftGeometry = None):
        self.geom = geom or AircraftGeometry()
        # Execution order matters: GroundEffect & Canard populate ctx before PolhamusWing runs
        self.modules = {
            "ground_effect": GroundEffectModule(),
            "canard": CanardModule(self.geom),
            "polhamus_wing": PolhamusWingModule(self.geom),
            "elevons": ElevonControlModule(self.geom),
            "propulsion_torque": PropulsionAndTorqueModule(self.geom),
            "tip_fins": TipFinAndFuselageModule(),
        }

    def set_enabled(self, module_name: str, enabled: bool):
        if module_name in self.modules:
            self.modules[module_name].enabled = enabled

    def evaluate(self, state: FlightState, ctrl: ControlInputs) -> tuple[AeroWrench, dict]:
        total = AeroWrench()
        ctx = {}
        for name, mod in self.modules.items():
            if mod.enabled:
                total += mod.compute(state, ctrl, self.geom, ctx)

        # Populate legacy diagnostic summary keys for Blender UI & Console reports
        ctx["CL"] = ctx.get("CL_wing", 0.0) + ctx.get("CL_can", 0.0) + ctx.get("CL_elev", 0.0)
        ctx["CD"] = ctx.get("CD_wing", 0.02) + ctx.get("CD_can", 0.0) + ctx.get("CD_elev", 0.0)
        qbar = 0.5 * self.geom.rho * (state.V ** 2)
        ctx["Cm"] = total.m / max(1e-4, qbar * self.geom.s_ref * self.geom.mac)
        return total, ctx