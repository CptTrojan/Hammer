"""
KITE Flight Sim — 6-DOF Modular Aerodynamics & Rigid-Body Simulator
===================================================================
Run inside Blender 5.x: Scripting -> Open -> Run Script

Features:
  * Full 6-DOF quaternion rigid-body integration (no gimbal lock)
  * Modular aerodynamic phenomena toggles (Canard, Motor Torque & Gyro, Ground Effect, Tip Fins)
  * Complete 3-axis rotation and lateral drift baking (Roll, Pitch, Yaw)
  * Live Radiomaster/UI elevon mixing and throttle-responsive propeller spin
"""

import bpy, math, os, sys, importlib
from mathutils import Vector
from bpy.props import FloatProperty, BoolProperty, PointerProperty
from bpy.types import PropertyGroup, Panel, Operator

# Ensure local directory is in path for module loading
sim_dir = os.path.dirname(os.path.abspath(__file__))
if sim_dir not in sys.path:
    sys.path.insert(0, sim_dir)

import kite_aero
import kite_dynamics
importlib.reload(kite_aero)
importlib.reload(kite_dynamics)

# Initialize simulator instance
SIM_ENGINE = kite_dynamics.SixDOFSimulator()
geom = SIM_ENGINE.geom


# ===================================================================
#  ENVIRONMENT — ground + sky + sun
# ===================================================================

def _switch_viewport(mode='RENDERED'):
    """Force all 3D viewports to Rendered mode (EEVEE) for full sky + ground."""
    try:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    except TypeError:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = mode


def setup_environment():
    """Create desert ground, sky, and sun light."""
    gnd_name = "KITE_Ground"
    if gnd_name not in bpy.data.objects:
        bpy.ops.mesh.primitive_plane_add(size=2000, location=(500, 0, 0))
        gnd = bpy.context.active_object
        gnd.name = gnd_name

        mat = bpy.data.materials.new("Ground_Desert")
        mat.use_nodes = True
        nt = mat.node_tree
        nodes = nt.nodes
        links = nt.links
        bsdf = nodes.get("Principled BSDF")

        bsdf.inputs['Base Color'].default_value = (0.04, 0.15, 0.02, 1)
        bsdf.inputs['Roughness'].default_value = 0.95
        try:
            bsdf.inputs['Specular IOR Level'].default_value = 0.1
        except Exception:
            pass

        noise = nodes.new('ShaderNodeTexNoise')
        noise.location = (-400, 200)
        noise.inputs['Scale'].default_value = 35.0
        noise.inputs['Detail'].default_value = 10.0

        ramp = nodes.new('ShaderNodeValToRGB')
        ramp.location = (-150, 200)
        els = ramp.color_ramp.elements
        els[0].color = (0.015, 0.08, 0.005, 1)
        els[0].position = 0.25
        els[1].color = (0.06, 0.22, 0.02, 1)
        els[1].position = 0.75
        links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
        links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
        mat.diffuse_color = (0.04, 0.15, 0.02, 1)
        gnd.data.materials.append(mat)

    sun_name = "KITE_Sun"
    if sun_name not in bpy.data.objects:
        sun_data = bpy.data.lights.new(sun_name, 'SUN')
        sun_data.energy = 5.0
        sun_data.color = (1.0, 0.92, 0.8)
        sun_obj = bpy.data.objects.new(sun_name, sun_data)
        bpy.context.collection.objects.link(sun_obj)
        sun_obj.rotation_euler = (math.radians(60), math.radians(10), math.radians(20))

    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("KITE_World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new('ShaderNodeOutputWorld')
    out.location = (400, 0)
    bg = nodes.new('ShaderNodeBackground')
    bg.location = (200, 0)
    bg.inputs['Strength'].default_value = 1.0
    links.new(bg.outputs['Background'], out.inputs['Surface'])

    try:
        sky = nodes.new('ShaderNodeTexSky')
        sky.location = (0, 0)
        sky.sky_type = 'HOSEK_WILKIE'
        sky.sun_direction = Vector((0.3, 0.2, 0.85)).normalized()
        sky.turbidity = 2.5
        links.new(sky.outputs['Color'], bg.inputs['Color'])
    except Exception:
        bg.inputs['Color'].default_value = (0.4, 0.6, 0.95, 1)

    _switch_viewport()
    print("[KITE] Environment: desert ground + sky + sun ready")


# ===================================================================
#  BLENDER ANIMATION & BAKING
# ===================================================================

def clear_sim():
    for nm in list(bpy.data.objects.keys()):
        if nm.startswith("FlightTrail") or nm.startswith("SimMarker"):
            bpy.data.objects.remove(bpy.data.objects[nm], do_unlink=True)
    for m in list(bpy.data.materials):
        if m.name.startswith("Trail_"):
            bpy.data.materials.remove(m)


def find_drone():
    for nm in ("KITE_Canard_Delta",):
        if nm in bpy.data.objects:
            return bpy.data.objects[nm]
    return None


def bake(frames):
    drone = find_drone()
    if not drone:
        print("ERROR: KITE_Canard_Delta not found")
        return
    sc = bpy.context.scene
    bpy.context.view_layer.update()

    MM = 0.001
    COCK_X = 600.0
    BUFFER_X = 2825.0
    STROKE = (BUFFER_X - COCK_X) * MM

    shuttle = bpy.data.objects.get("Shuttle")
    rail_frame = bpy.data.objects.get("Rail_Frame")
    propeller = bpy.data.objects.get("Propeller")

    rail_rad = frames[0]["theta"]
    launch_v = frames[0]["V"]
    launch_h = frames[0]["z"]

    t_rail = 2.0 * STROKE / launch_v if launch_v > 0.01 else 1.0
    fps = 24
    n_hold = 6
    n_rail = max(6, int(t_rail * fps))
    frame_offset = n_hold + n_rail

    sc.frame_start = 1
    sc.frame_end = frame_offset + frames[-1]["frame"]
    sc.render.fps = fps

    clear_sim()
    print(f"[KITE] Baking 6-DOF: {n_hold} hold + {n_rail} rail + {len(frames)} flight frames")

    dg = bpy.context.evaluated_depsgraph_get()
    drone_eval = drone.evaluated_get(dg)
    pos_cock = drone_eval.matrix_world.translation.copy()
    actual_pitch = drone_eval.matrix_world.to_euler('XYZ')[1]

    if rail_frame:
        rail_eval = rail_frame.evaluated_get(dg)
        rail_fwd = rail_eval.matrix_world.to_3x3() @ Vector((1, 0, 0))
    else:
        rail_fwd = Vector((math.cos(rail_rad), 0, math.sin(rail_rad)))
    pos_buffer = pos_cock + rail_fwd * STROKE

    if drone.animation_data:
        drone.animation_data_clear()
    if drone.parent:
        world_mat = drone.matrix_world.copy()
        drone.parent = None
        drone.matrix_world = world_mat

    drone.rotation_mode = 'XYZ'
    if not drone.animation_data:
        drone.animation_data_create()

    action = bpy.data.actions.new("KITE_FlightAnim")
    slot = action.slots.new(id_type='OBJECT', name="KITE_Flight")
    layer = action.layers.new("FlightLayer")
    strip = layer.strips.new(type='KEYFRAME')
    cbag = strip.channelbag(slot, ensure=True)

    fc_lx = cbag.fcurves.new("location", index=0)
    fc_ly = cbag.fcurves.new("location", index=1)
    fc_lz = cbag.fcurves.new("location", index=2)
    fc_rx = cbag.fcurves.new("rotation_euler", index=0)  # Roll
    fc_ry = cbag.fcurves.new("rotation_euler", index=1)  # Pitch
    fc_rz = cbag.fcurves.new("rotation_euler", index=2)  # Yaw

    co_lx, co_ly, co_lz = [], [], []
    co_rx, co_ry_l, co_rz = [], [], []

    # Hold phase
    for i in range(n_hold):
        fr = float(i + 1)
        co_lx.extend([fr, pos_cock.x])
        co_ly.extend([fr, pos_cock.y])
        co_lz.extend([fr, pos_cock.z])
        co_rx.extend([fr, 0.0])
        co_ry.extend([fr, actual_pitch])
        co_rz.extend([fr, 0.0])

    # Rail acceleration phase
    for i in range(n_rail):
        fr = float(n_hold + i + 1)
        frac = ((i + 1) / n_rail) ** 2
        wx = pos_cock.x + (pos_buffer.x - pos_cock.x) * frac
        wy = pos_cock.y + (pos_buffer.y - pos_cock.y) * frac
        wz = pos_cock.z + (pos_buffer.z - pos_cock.z) * frac
        co_lx.extend([fr, wx])
        co_ly.extend([fr, wy])
        co_lz.extend([fr, wz])
        co_rx.extend([fr, 0.0])
        co_ry.extend([fr, actual_pitch])
        co_rz.extend([fr, 0.0])

    # 6-DOF Free Flight Phase
    crashed = frames[-1]["crashed"]
    for f in frames:
        fr = float(frame_offset + f["frame"])
        co_lx.extend([fr, pos_buffer.x + f["x"]])
        co_ly.extend([fr, pos_buffer.y - f["y"]])  # Blender +Y is Left, Body +Y is Right
        co_lz.extend([fr, pos_buffer.z + (f["z"] - launch_h)])
        co_rx.extend([fr, f["phi"]])
        co_ry.extend([fr, -f["theta"]])
        co_rz.extend([fr, -f["psi"]])

    n_crash = 0
    crash_frame = 0
    if crashed:
        last = frames[-1]
        impact_x = pos_buffer.x + last["x"]
        impact_y = pos_buffer.y - last["y"]
        ground_z = 0.02
        impact_v = last["V"]
        impact_theta = last["theta"]

        n_crash = 6
        crash_frame = frame_offset + last["frame"]
        crash_start_fr = crash_frame

        for i in range(1, n_crash + 1):
            fr = float(crash_start_fr + i)
            t = i / n_crash
            dx = impact_v * 0.05 * t
            pitch = impact_theta + math.radians(90) * t
            co_lx.extend([fr, impact_x + dx])
            co_ly.extend([fr, impact_y])
            co_lz.extend([fr, ground_z])
            co_rx.extend([fr, 0.0])
            co_ry.extend([fr, -pitch])
            co_rz.extend([fr, -last["psi"]])

    total_n = n_hold + n_rail + len(frames) + n_crash
    sc.frame_end = int(co_lx[-2]) + 1 if co_lx else 1

    for fc, co in ((fc_lx, co_lx), (fc_ly, co_ly), (fc_lz, co_lz),
                    (fc_rx, co_rx), (fc_ry, co_ry_l), (fc_rz, co_rz)):
        fc.keyframe_points.add(count=total_n)
        fc.keyframe_points.foreach_set("co", co)
        fc.update()
        fc.extrapolation = 'CONSTANT'

    drone.animation_data.action = action
    drone.animation_data.action_slot = slot

    # Shuttle animation
    if shuttle:
        if shuttle.animation_data:
            shuttle.animation_data_clear()
        if not shuttle.animation_data:
            shuttle.animation_data_create()

        s_action = bpy.data.actions.new("KITE_ShuttleAnim")
        s_slot = s_action.slots.new(id_type='OBJECT', name="ShuttleSlot")
        s_layer = s_action.layers.new("ShuttleLayer")
        s_strip = s_layer.strips.new(type='KEYFRAME')
        s_cbag = s_strip.channelbag(s_slot, ensure=True)
        s_fc = s_cbag.fcurves.new("location", index=0)

        s_co = []
        for i in range(n_hold):
            s_co.extend([float(i + 1), COCK_X * MM])
        for i in range(n_rail):
            frac = ((i + 1) / n_rail) ** 2
            sx = (COCK_X + (BUFFER_X - COCK_X) * frac) * MM
            s_co.extend([float(n_hold + i + 1), sx])
        s_co.extend([float(frame_offset + 1), BUFFER_X * MM])
        s_co.extend([float(sc.frame_end), BUFFER_X * MM])
        n_shuttle = len(s_co) // 2

        s_fc.keyframe_points.add(count=n_shuttle)
        s_fc.keyframe_points.foreach_set("co", s_co)
        s_fc.update()
        s_fc.extrapolation = 'CONSTANT'
        shuttle.animation_data.action = s_action
        shuttle.animation_data.action_slot = s_slot

    # Propeller spin animation
    if propeller:
        if propeller.animation_data:
            propeller.animation_data_clear()
        if not propeller.animation_data:
            propeller.animation_data_create()

        p_action = bpy.data.actions.new("KITE_PropAnim")
        p_slot = p_action.slots.new(id_type='OBJECT', name="PropSlot")
        p_layer = p_action.layers.new("PropLayer")
        p_strip = p_layer.strips.new(type='KEYFRAME')
        p_cbag = p_strip.channelbag(p_slot, ensure=True)
        p_fc = p_cbag.fcurves.new("rotation_euler", index=0)

        crash_frame_num = frame_offset + frames[-1]["frame"] if crashed else sc.frame_end
        p_co = []
        last_angle = 0.0
        for i in range(1, sc.frame_end + 1):
            if i <= crash_frame_num:
                last_angle = math.radians(37 * i)
            p_co.extend([float(i), last_angle])
        n_prop = len(p_co) // 2
        p_fc.keyframe_points.add(count=n_prop)
        p_fc.keyframe_points.foreach_set("co", p_co)
        p_fc.update()
        p_fc.extrapolation = 'CONSTANT'
        propeller.animation_data.action = p_action
        propeller.animation_data.action_slot = p_slot

    # Flight path trail curve
    cd = bpy.data.curves.new("FlightTrail_C", 'CURVE')
    cd.dimensions = '3D'
    sp = cd.splines.new('POLY')
    sp.points.add(len(frames) - 1)
    for i, f in enumerate(frames):
        sp.points[i].co = (pos_buffer.x + f["x"], pos_buffer.y - f["y"],
                           pos_buffer.z + (f["z"] - launch_h), 1)

    trail = bpy.data.objects.new("FlightTrail", cd)
    bpy.context.collection.objects.link(trail)
    cd.bevel_depth = 0.005
    mat_t = bpy.data.materials.new("Trail_Green")
    mat_t.diffuse_color = (0.1, 0.85, 0.2, 1)
    trail.data.materials.append(mat_t)

    sc.frame_set(1)
    _switch_viewport()

    max_alt = max(f["z"] for f in frames)
    max_v = max(f["V"] for f in frames)
    t_end = frames[-1]["t"]
    print("\n" + "=" * 65)
    print("  KITE 6-DOF FLIGHT SIMULATION COMPLETED")
    print("=" * 65)
    print(f"  Launch V   : {launch_v:.1f} m/s")
    print(f"  Max alt    : {max_alt:.2f} m")
    print(f"  Max speed  : {max_v:.1f} m/s ({max_v*3.6:.0f} km/h)")
    print(f"  Flight time: {t_end:.2f} s")
    print(f"  Outcome    : {'CRASH' if crashed else 'FLYING'}")
    print("=" * 65 + "\n")


# ===================================================================
#  UI PROPERTIES & LIVE CALLBACKS
# ===================================================================

def _update_elevon(self, context):
    deg = self.elevon
    rad = math.radians(deg)
    for name, sign in (("Elevon_L", 1), ("Elevon_R", 1)):
        ob = bpy.data.objects.get(name)
        if ob:
            ob.rotation_mode = 'XYZ'
            ob.rotation_euler[1] = sign * rad


_prop_spin_active = False
_prop_spin_angle = 0.0

def _prop_spin_tick():
    global _prop_spin_angle, _prop_spin_active
    prop = bpy.data.objects.get("Propeller")
    if not prop or not _prop_spin_active:
        _prop_spin_active = False
        return None

    try:
        thr = bpy.context.scene.kite_sim.throttle
    except Exception:
        _prop_spin_active = False
        return None

    rps = (thr / 100.0) * 130.0
    dt = 0.03
    _prop_spin_angle += rps * 2 * math.pi * dt
    prop.rotation_mode = 'XYZ'
    prop.rotation_euler[0] = _prop_spin_angle

    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()
    return dt


def _update_throttle(self, context):
    global _prop_spin_active, _prop_spin_angle
    prop = bpy.data.objects.get("Propeller")
    if not prop:
        return
    if self.throttle > 0.1:
        if not _prop_spin_active:
            _prop_spin_active = True
            _prop_spin_angle = prop.rotation_euler[0] if prop.rotation_mode == 'XYZ' else 0.0
            bpy.app.timers.register(_prop_spin_tick, first_interval=0.03)
    else:
        _prop_spin_active = False
        prop.rotation_mode = 'XYZ'
        prop.rotation_euler[0] = 0.0


def _rebuild_hydraulic(th_rad):
    MM = 0.001
    xa = 2100.0
    pin_h = 40.0
    outer_h = 50.0
    inner_r = 28.0
    rod_r = 14.0

    top_z = pin_h + xa * math.sin(th_rad) - 20 * math.cos(th_rad)
    top_z = max(top_z, outer_h + 10)

    outer = bpy.data.objects.get("Hydraulic_Outer")
    if not outer:
        return
    foot_x = outer.location.x / MM
    root = bpy.data.objects.get("KITE_Catapult")

    for nm in ("Hydraulic_Inner", "Hydraulic_Rod"):
        ob = bpy.data.objects.get(nm)
        if ob:
            bpy.data.objects.remove(ob, do_unlink=True)

    m_alu = bpy.data.materials.get("Rail_Alu")
    m_steel = bpy.data.materials.get("Steel")

    inner_h = top_z - 10
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=32, radius=inner_r * MM, depth=inner_h * MM,
        location=(foot_x * MM, 0, (10 + inner_h / 2) * MM))
    inner = bpy.context.object
    inner.name = "Hydraulic_Inner"
    if root:
        inner.parent = root
    if m_alu:
        inner.data.materials.append(m_alu)

    rod_h = top_z - outer_h
    if rod_h > 5:
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32, radius=rod_r * MM, depth=rod_h * MM,
            location=(foot_x * MM, 0, (outer_h + rod_h / 2) * MM))
        rod = bpy.context.object
        rod.name = "Hydraulic_Rod"
        if root:
            rod.parent = root
        if m_steel:
            rod.data.materials.append(m_steel)


def _update_rail_angle(self, context):
    rail = bpy.data.objects.get("Rail_Frame")
    if not rail:
        return
    th = math.radians(self.rail_angle)
    rail.rotation_mode = 'XYZ'
    rail.rotation_euler[1] = -th
    _rebuild_hydraulic(th)

    drone = bpy.data.objects.get("KITE_Canard_Delta")
    if drone:
        MM = 0.001
        if drone.animation_data:
            drone.animation_data_clear()
        if drone.parent != rail:
            drone.parent = rail
        drone.matrix_parent_inverse.identity()
        drone.location = Vector(((600.0 + 205.0) * MM, 0, (100.0 + 150.0) * MM))
        drone.rotation_euler = (0, 0, 0)
    _switch_viewport()


class KiteSimProps(PropertyGroup):
    launch_velocity: FloatProperty(
        name="Launch Velocity (m/s)", default=12.0, min=2.0, max=30.0, precision=1
    )
    rail_angle: FloatProperty(
        name="Rail Angle (deg)", default=15.0, min=0.0, max=45.0, precision=1,
        update=_update_rail_angle
    )
    throttle: FloatProperty(
        name="Throttle (%)", default=100.0, min=0.0, max=100.0, precision=0,
        update=_update_throttle
    )
    elevon: FloatProperty(
        name="Elevon Pitch (deg)", default=0.0, min=-20.0, max=20.0, precision=1,
        update=_update_elevon
    )
    roll_cmd: FloatProperty(
        name="Roll / Aileron (deg)", default=0.0, min=-18.0, max=18.0, precision=1
    )
    sim_time: FloatProperty(
        name="Duration (s)", default=10.0, min=1.0, max=60.0, precision=1
    )
    launch_height: FloatProperty(
        name="Launch Height (m)", default=1.0, min=0.3, max=5.0, precision=1
    )
    
    # Phenomena Toggles
    use_canard: BoolProperty(name="Canard & Downwash", default=True)
    use_torque: BoolProperty(name="Motor Torque & Gyro", default=True)
    use_ground_effect: BoolProperty(name="Ground Effect", default=True)
    use_tip_fins: BoolProperty(name="Tip Fins & Sideslip", default=True)


# ===================================================================
#  OPERATORS
# ===================================================================

class KITE_OT_SetSim(Operator):
    bl_idname = "kite.set_sim"
    bl_label = "Set Simulation"
    bl_description = "Compute 6-DOF flight path, bake keyframes, show trail"

    def execute(self, context):
        global _prop_spin_active
        _prop_spin_active = False
        bpy.ops.kite.reset_sim()

        p = context.scene.kite_sim

        # Configure phenomena modules
        SIM_ENGINE.aero.set_enabled("canard", p.use_canard)
        SIM_ENGINE.aero.set_enabled("propulsion_torque", p.use_torque)
        SIM_ENGINE.aero.set_enabled("ground_effect", p.use_ground_effect)
        SIM_ENGINE.aero.set_enabled("tip_fins", p.use_tip_fins)

        drone_obj = find_drone()
        rail_obj = bpy.data.objects.get("Rail_Frame")
        if drone_obj and rail_obj:
            dg = bpy.context.evaluated_depsgraph_get()
            cock_z = drone_obj.evaluated_get(dg).matrix_world.translation.z
            rfwd = rail_obj.evaluated_get(dg).matrix_world.to_3x3() @ Vector((1, 0, 0))
            STROKE = 2.225
            actual_launch_h = cock_z + rfwd.z * STROKE
        else:
            actual_launch_h = p.launch_height

        frames = SIM_ENGINE.simulate_trajectory(
            launch_v=p.launch_velocity,
            rail_deg=p.rail_angle,
            throttle_pct=p.throttle,
            elevon_deg=p.elevon,
            roll_deg=p.roll_cmd,
            sim_time=p.sim_time,
            launch_h=actual_launch_h,
            fps=context.scene.render.fps
        )
        bake(frames)
        context.scene.frame_set(1)

        outcome = 'CRASH' if frames[-1]['crashed'] else 'FLYING'
        max_alt = max(f['z'] for f in frames)
        self.report({'INFO'}, f"{outcome} | max alt {max_alt:.1f}m")
        _switch_viewport()
        return {'FINISHED'}


class KITE_OT_PlaySim(Operator):
    bl_idname = "kite.play_sim"
    bl_label = "Play"
    bl_description = "Play the baked flight animation with tracking camera"

    def execute(self, context):
        sc = context.scene
        drone = find_drone()
        cam_name = "KITE_Camera"
        cam_obj = bpy.data.objects.get(cam_name)
        if not cam_obj:
            cam_data = bpy.data.cameras.new(cam_name)
            cam_data.lens = 50
            cam_data.clip_end = 5000
            cam_obj = bpy.data.objects.new(cam_name, cam_data)
            bpy.context.collection.objects.link(cam_obj)

        cam_obj.location = (-3, -6, 3)
        cam_obj.rotation_euler = (math.radians(70), 0, math.radians(-25))

        if drone:
            for c in list(cam_obj.constraints):
                cam_obj.constraints.remove(c)
            con = cam_obj.constraints.new('TRACK_TO')
            con.target = drone
            con.track_axis = 'TRACK_NEGATIVE_Z'
            con.up_axis = 'UP_Y'

        sc.camera = cam_obj
        for area in context.window.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.region_3d.view_perspective = 'CAMERA'
                break

        sc.frame_set(sc.frame_start)
        bpy.ops.screen.animation_cancel(restore_frame=False)
        bpy.ops.screen.animation_play()
        _switch_viewport()
        self.report({'INFO'}, "Playing with tracking camera...")
        return {'FINISHED'}


class KITE_OT_ResetSim(Operator):
    bl_idname = "kite.reset_sim"
    bl_label = "Reset"
    bl_description = "Remove flight animation and rebuild drone on catapult"

    def execute(self, context):
        global _prop_spin_active
        _prop_spin_active = False
        try:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        except Exception:
            pass

        clear_sim()

        parts_to_delete = [
            "KITE_Canard_Delta", "Fuselage", "Wing", "Canards",
            "Elevon_L", "Elevon_R", "TipFin_L", "TipFin_R",
            "Motor_Pylon", "Motor", "Prop_Shaft", "Propeller", "Prop_Hub",
            "CG_Marker", "Prop_Disc_Clearance",
        ]
        for nm in parts_to_delete:
            ob = bpy.data.objects.get(nm)
            if ob:
                bpy.data.objects.remove(ob, do_unlink=True)

        shuttle = bpy.data.objects.get("Shuttle")
        if shuttle and shuttle.animation_data:
            shuttle.animation_data_clear()
            shuttle.location.x = 0.600

        rail = bpy.data.objects.get("Rail_Frame")
        if rail:
            try:
                import importlib.util
                drone_path = os.path.join(sim_dir, "kite_canard_delta.py")
                spec = importlib.util.spec_from_file_location("kite_canard_delta", drone_path)
                kcd = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(kcd)

                droot = kcd.build(clear=False)
                MM = 0.001
                bpy.context.view_layer.update()
                droot.parent = rail
                droot.matrix_parent_inverse.identity()
                droot.location = Vector(((600.0 + 205.0) * MM, 0, (100.0 + 150.0) * MM))
                droot.rotation_euler = (0, 0, 0)
                bpy.context.view_layer.update()
            except Exception as e:
                print(f"[KITE] Reset rebuild error: {e}")

        context.scene.frame_set(1)
        _switch_viewport()
        self.report({'INFO'}, "Reset — drone rebuilt on catapult")
        return {'FINISHED'}


# ===================================================================
#  PANEL
# ===================================================================

class KITE_PT_SimPanel(Panel):
    bl_label = "KITE 6-DOF Flight Sim"
    bl_idname = "KITE_PT_sim_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "KITE"

    def draw(self, context):
        ly = self.layout
        p = context.scene.kite_sim

        b = ly.box()
        b.label(text="6-DOF Aerodynamics & Stability", icon='FORCE_WIND')
        c = b.column(align=True)
        c.label(text=f"Span {geom.span*1e3:.0f}mm | Sref {geom.s_ref:.4f}m\u00