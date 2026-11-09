"""
KITE bungee rail catapult (3 m) - parametric Blender model + launch animation
===========================================================================
Run inside Blender (Scripting tab -> Run Script) or headless:
    blender --background --python kite_catapult.py
Keep kite_canard_delta.py in the same folder: the drone is loaded from it
and placed on the shuttle.  All dimensions in mm.  World: +X = launch
direction, +Z up, origin = rear ground pivot.

Concept
  * 3 x 1 m aluminium box rail (80 x 100), bolted splice plates underneath
  * rear ground plate with pivot, A-frame on wheels sets the launch angle
  * shuttle (carriage) on UHMW pads; drone sits in two contoured saddles
    and is pushed by rubber pads against the canard trailing-edge roots
  * twin bungee: anchored under the rear, runs forward under the rail,
    round a front sheave and back along the top to the shuttle hook
  * cock: hand winch + strap pulls shuttle back onto the release latch,
    unhook strap, pull the red release lever -> shuttle runs ~2.2 m and
    stops on front rubber buffers; drone flies off
  * cradle height chosen so the pusher prop disc clears the rail/latch
"""
import sys, os, math, bpy
from mathutils import Vector, Matrix

import importlib.util

drone_path = r"C:\Users\adity\Desktop\Hammer\Hammer\kite_canard_delta.py"
spec = importlib.util.spec_from_file_location("kite_canard_delta", drone_path)
drone = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drone)

# ----------------------------------------------------------------- PARAMETERS
C = dict(
    rail_len=3000.0, rail_w=80.0, rail_h=100.0, n_sections=3,
    angle_deg=15.0,            # launch angle
    pin_h=40.0,                # rear pivot height above ground
    aframe_at=2100.0,          # A-frame position along rail
    aframe_spread=900.0,       # wheel track
    wheel_d=150.0,
    shuttle_len=330.0,         # base plate length
    cradle_h=150.0,            # rail top -> fuselage axis
    prop_clear_min=25.0,       # required gap prop tip -> rail top
    cock_x=600.0,              # shuttle reference (push-pad face) when cocked
    buffer_x=2825.0,           # front buffer face
    sheave_x=3075.0, sheave_d=125.0,
    bungee_d=10.0, bungee_y=15.0,
    # animation (slow motion: real stroke ~0.15 s shown over 36 frames)
    f_cocked=12, f_release_len=36, f_end=96,
)
MM = 0.001
P = drone.P
CANARD_TE = P["can_le"] + P["can_root_chord"]               # push station
PROP_S = P["pylon_s1"] + P["motor_len"] + 10                 # prop plane station
PROP_Z = P["wing_z"] + P["motor_axis_z"]                     # prop axis vs fuselage axis
TOP = C["rail_h"]

# ------------------------------------------------------------------ helpers
def v(x, y, z):
    return Vector((x * MM, y * MM, z * MM))

def mesh_obj(name, verts_mm, faces, m, parent, smooth=False):
    me = bpy.data.meshes.new(name)
    me.from_pydata([Vector(p) * MM for p in verts_mm], [], faces)
    me.validate(); me.update()
    import bmesh
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(m)
    ob.parent = parent
    if smooth:
        for p in me.polygons: p.use_smooth = True
    return ob

def box(name, x0, x1, y0, y1, z0, z1, m, parent):
    vs = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    fs = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return mesh_obj(name, vs, fs, m, parent)

def oriented(ob, p0, p1):
    d = Vector(p1) - Vector(p0)
    ob.rotation_mode = 'QUATERNION'
    ob.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d.normalized())
    ob.location = (Vector(p0) + Vector(p1)) / 2 * MM

def cyl(name, p0, p1, r, m, parent, n=32):
    L = (Vector(p1) - Vector(p0)).length
    bpy.ops.mesh.primitive_cylinder_add(vertices=n, radius=r * MM, depth=L * MM)
    ob = bpy.context.object; ob.name = name
    ob.data.materials.append(m); ob.parent = parent
    oriented(ob, p0, p1)
    return ob

def beam(name, p0, p1, w, h, m, parent):
    L = (Vector(p1) - Vector(p0)).length
    ob = box(name, -w / 2, w / 2, -h / 2, h / 2, -L / 2, L / 2, m, parent)
    oriented(ob, p0, p1)
    return ob

def empty(name, parent, loc_mm=(0, 0, 0), rot=(0, 0, 0)):
    e = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(e)
    e.parent = parent; e.location = v(*loc_mm); e.rotation_euler = rot
    e.empty_display_size = 0.05
    return e

def apply_bool(target, cutter):
    for o in bpy.context.selected_objects: o.select_set(False)
    bpy.context.view_layer.objects.active = target; target.select_set(True)
    mod = target.modifiers.new("cut", 'BOOLEAN')
    mod.object = cutter; mod.operation = 'DIFFERENCE'; mod.solver = 'EXACT'
    bpy.ops.object.modifier_apply(modifier=mod.name)

# --------------------------------------------------------------- build parts
def build_rail(rail, M):
    W, H, L = C["rail_w"], C["rail_h"], C["rail_len"]
    hw = W / 2
    prof = [(-hw, 0), (hw, 0), (hw, 45), (hw - 6, 45), (hw - 6, 55), (hw, 55),
            (hw, H), (-hw, H), (-hw, 55), (-hw + 6, 55), (-hw + 6, 45), (-hw, 45)]
    n = len(prof)
    seg = L / C["n_sections"]
    for i in range(C["n_sections"]):
        x0, x1 = i * seg + (0.5 if i else 0), (i + 1) * seg - (0.5 if i < C["n_sections"] - 1 else 0)
        vs = [(x0, y, z) for y, z in prof] + [(x1, y, z) for y, z in prof]
        fs = [(j, (j + 1) % n, n + (j + 1) % n, n + j) for j in range(n)]
        fs += [tuple(reversed(range(n))), tuple(range(n, 2 * n))]
        mesh_obj(f"Rail_Section_{i+1}", vs, fs, M["alu"], rail)
        if i:                               # splice plate + bolts under joint
            xj = i * seg
            box(f"Splice_{i}", xj - 100, xj + 100, -32, 32, -6, 0, M["steel"], rail)
            for dx in (-70, -30, 30, 70):
                for dy in (-18, 18):
                    cyl(f"Bolt_{i}", (xj + dx, dy, -12), (xj + dx, dy, -6), 6, M["steel"], rail, 6)
    box("EndCap_Rear", -6, 0, -hw, hw, 0, H, M["black"], rail)
    box("EndCap_Front", L, L + 6, -hw, hw, 0, H, M["black"], rail)

    # front sheave
    sx, r = C["sheave_x"], C["sheave_d"] / 2
    sz = TOP + 5 - r
    for sgn in (1, -1):
        box("Sheave_Bracket", L - 60, sx + 40, sgn * hw, sgn * (hw + 6), sz - 30, TOP, M["steel"], rail)
    cyl("Sheave", (sx, -20, sz), (sx, 20, sz), r, M["uhmw"], rail, 48)
    cyl("Sheave_Axle", (sx, -52, sz), (sx, 52, sz), 6, M["steel"], rail)

    # front buffers
    for sgn in (1, -1):
        box("Buffer", C["buffer_x"], C["buffer_x"] + 60, sgn * 26, sgn * 40, TOP, TOP + 40, M["rubber"], rail)

    # bungee lower run (fixed) + rear anchor
    by, bz = C["bungee_y"], sz - r
    for sgn in (1, -1):
        cyl("Bungee_Lower", (250, sgn * by, bz), (sx, sgn * by, bz), C["bungee_d"] / 2, M["bungee"], rail, 12)
    box("Bungee_Anchor", 230, 270, -30, 30, bz - 10, 0, M["steel"], rail)
    return sz

def build_rear(root, rail, M):
    ph = C["pin_h"]
    box("Ground_Plate", -150, 250, -130, 130, 0, 6, M["steel"], root)
    for x in (-120, 220):
        for y in (-100, 100):
            cyl("Stake_Hole", (x, y, 5.9), (x, y, 6.5), 9, M["black"], root, 16)
    for sgn in (1, -1):
        box("Pivot_Lug", -30, 30, sgn * 41, sgn * 51, 6, ph + 25, M["steel"], root)
    cyl("Pivot_Pin", (0, -58, ph), (0, 58, ph), 8, M["steel"], root)

    # release latch (rail frame) - claw holds shuttle pin
    lx = C["cock_x"] - C["shuttle_len"] + 40 - 5     # pin position
    box("Latch_Base", lx - 50, lx + 2, -18, 18, TOP, TOP + 12, M["steel"], rail)
    box("Latch_Claw", lx - 9, lx + 9, -8, 8, TOP + 2, TOP + 16, M["steel"], rail)
    cyl("Lever_Pivot", (lx - 25, 18, TOP + 8), (lx - 25, 54, TOP + 8), 5, M["steel"], rail)
    beam("Release_Lever", (lx - 25, 52, TOP + 8), (lx - 95, 52, TOP + 110), 10, 6, M["red"], rail)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=11 * MM, location=v(lx - 95, 52, TOP + 110))
    k = bpy.context.object; k.name = "Release_Knob"; k.parent = rail; k.data.materials.append(M["red"])
    bpy.ops.mesh.primitive_torus_add(major_radius=14 * MM, minor_radius=2 * MM,
                                     location=v(lx - 100, 52, TOP + 128), rotation=(math.pi / 2, 0, 0))
    t = bpy.context.object; t.name = "Pull_Cord_Ring"; t.parent = rail; t.data.materials.append(M["red"])
    beam("Safety_Lever", (lx - 60, 52, TOP + 8), (lx - 80, 52, TOP + 75), 8, 5, M["red"], rail)
    cyl("Safety_Pin", (lx - 60, -24, TOP + 8), (lx - 60, 58, TOP + 8), 3.5, M["red"], rail, 12)

    # cocking winch on left side
    wx = 160
    box("Winch_Plate", wx - 40, wx + 40, 40, 46, 20, 90, M["steel"], rail)
    cyl("Winch_Drum", (wx, 46, 55), (wx, 86, 55), 18, M["black"], rail)
    cyl("Winch_Flange", (wx, 84, 55), (wx, 90, 55), 28, M["steel"], rail)
    beam("Winch_Crank", (wx, 90, 55), (wx - 70, 96, 55), 10, 6, M["steel"], rail)
    cyl("Winch_Handle", (wx - 70, 96, 55), (wx - 70, 130, 55), 6, M["red"], rail, 16)
    return lx

def build_hydraulic(root, M, rail_mat):
    """Telescopic hydraulic pole: outer tube on ground, inner tube + rod
    stretch up to the rail.  The inner/rod are rebuilt by the flight-sim
    rail-angle callback so they always match the current angle."""
    xa = C["aframe_at"]            # 2100 mm along rail
    W = rail_mat["W"]

    # rail attachment point (rail-local coords)
    attach_rail = (xa, 0, -20)
    top_world = W @ v(*attach_rail)
    top_mm = Vector(top_world) / MM

    # foot: directly below attachment, on the ground
    foot_x = top_mm.x
    foot = Vector((foot_x, 0, 0))

    # ── outer cylinder (short base, fixed on ground) ──
    outer_r = 35.0
    outer_h = 50.0
    cyl("Hydraulic_Outer", foot, foot + Vector((0, 0, outer_h)), outer_r, M["steel"], root)

    # foot plate
    box("Hydraulic_Foot", foot_x - 80, foot_x + 80, -60, 60, 0, 8, M["steel"], root)

    # seal ring on top of outer tube
    cyl("Hydraulic_Seal", foot + Vector((0, 0, outer_h - 5)),
        foot + Vector((0, 0, outer_h + 5)), outer_r + 4, M["steel"], root)

    # ── inner tube: vertical, from ground up to rail height ──
    inner_r = 28.0
    inner_h = top_mm.z - 10   # height from 10mm above ground
    cyl("Hydraulic_Inner",
        foot + Vector((0, 0, 10)),
        foot + Vector((0, 0, 10 + inner_h)),
        inner_r, M["alu"], root)

    # ── piston rod: vertical, from outer top to rail height ──
    rod_r = 14.0
    rod_h = top_mm.z - outer_h
    if rod_h > 5:
        cyl("Hydraulic_Rod",
            foot + Vector((0, 0, outer_h)),
            foot + Vector((0, 0, outer_h + rod_h)),
            rod_r, M["steel"], root)

    # clevis bracket at rail attachment
    box("Hydraulic_Clevis", xa - 30, xa + 30, -35, 35, -20, 0, M["steel"], rail_mat["rail"])

def build_shuttle(shuttle, M):
    L, H = C["shuttle_len"], C["cradle_h"]
    x0, x1 = -L + 40, 40
    box("Shuttle_Base", x0, x1, -80, 80, 8, 16, M["alu"], shuttle)
    for xa, xb in ((x0, x0 + 60), (x1 - 60, x1)):
        for sgn in (1, -1):
            box("UHMW_Pad", xa, xb, sgn * 10, sgn * 38, 0, 8, M["uhmw"], shuttle)
    for sgn in (1, -1):
        box("Shuttle_Skirt", x0, x1, sgn * 42, sgn * 48, -45, 8, M["alu"], shuttle)
        box("UHMW_Guide", x0, x1, sgn * 40.5, sgn * 42, -40, 0, M["uhmw"], shuttle)
        box("Latch_Lug", x0 - 10, x0 + 8, sgn * 25, sgn * 31, 0, 16, M["steel"], shuttle)
        box("Bungee_Hook", x1 - 14, x1, sgn * 8, sgn * 22, 0, 8, M["steel"], shuttle)
    cyl("Latch_Pin", (x0 - 5, -31, 4), (x0 - 5, 31, 4), 4, M["steel"], shuttle, 16)
    # strap hook for the winch
    box("Strap_Hook", x0, x0 + 20, 48, 54, -30, 0, M["steel"], shuttle)

    return x0, x1

def build_saddles(shuttle, M, cutters):
    """Z-bracket cradles under canard and wing.
    Foot bolted to shuttle base → vertical post → rubber pad under surface.
    Centre is fully open (280mm+ gap) — prop clears easily."""
    H = C["cradle_h"]
    prop_half = P["prop_d"] / 2 + 15        # 142mm half-gap
    tw = 10.0                                # bracket metal thickness
    pad_t = 8.0                              # rubber pad thickness
    base_top = 16.0                          # top of shuttle base plate

    # wing bottom surface Z (shuttle-local): H + wing_z - half_thickness
    # wing at station 420: ~11mm half-thickness at that chord position
    wing_pad_top = H + P["wing_z"] - 12.0   # = 150 + (-6) - 12 = 132
    # canard bottom Z: H + can_z - half_thickness (~5mm)
    can_pad_top = H + P["can_z"] - 6.0      # = 150 + 0 - 6 = 144
    out = []

    def make_bracket(name, lx, y_inner, y_outer, length, pad_top_z):
        """Z-bracket: foot on base → post → rubber pad just under wing/canard."""
        s = 1 if y_inner > 0 else -1
        hlf = length / 2

        # 1. foot: flat plate ON the shuttle base, extends 100mm inward to reach base
        foot_y0 = y_inner - s * 100          # 100mm inward from post (reaches shuttle base)
        foot_y1 = y_inner + s * tw           # flush with post outer face
        fy0, fy1 = min(foot_y0, foot_y1), max(foot_y0, foot_y1)
        box(f"{name}_Foot", lx - hlf, lx + hlf,
            fy0, fy1, base_top, base_top + tw, M["alu"], shuttle)

        # 2. vertical post: from foot top to pad bottom
        py0, py1 = min(y_inner, y_inner + s * tw), max(y_inner, y_inner + s * tw)
        box(f"{name}_Post", lx - hlf, lx + hlf,
            py0, py1, base_top + tw, pad_top_z - pad_t, M["alu"], shuttle)

        # 3. rubber pad: sits UNDER the wing/canard surface, not through it
        pad_y0, pad_y1 = min(y_inner, y_outer), max(y_inner, y_outer)
        box(f"{name}_Pad", lx - hlf, lx + hlf,
            pad_y0, pad_y1, pad_top_z - pad_t, pad_top_z, M["rubber"], shuttle)

    # canard cradle — station 180, pads stay within canard span
    can_st = 180.0
    can_lx = CANARD_TE - can_st
    can_hw = P["can_semispan"] - 20          # 173mm, well within canard tips
    for sgn, tag in ((1, "L"), (-1, "R")):
        make_bracket(f"Cradle_Canard_{tag}", can_lx,
                     sgn * prop_half, sgn * can_hw, 50.0, can_pad_top)
    out.append(("Cradle_Canard", can_st, can_pad_top))

    # wing cradle — station 480, pads stay within wing edge at that station
    wing_st = 480.0
    wing_lx = CANARD_TE - wing_st
    wing_hw = 180.0                          # well within wing edge (~216mm at this station)
    for sgn, tag in ((1, "L"), (-1, "R")):
        make_bracket(f"Cradle_Wing_{tag}", wing_lx,
                     sgn * prop_half, sgn * wing_hw, 60.0, wing_pad_top)
    out.append(("Cradle_Wing", wing_st, wing_pad_top))

    return out

# -------------------------------------------------------------------- main
def build():
    drone.clear_scene()
    sc = bpy.context.scene
    sc.unit_settings.system = 'METRIC'; sc.unit_settings.length_unit = 'MILLIMETERS'
    mk = drone.mat
    M = dict(alu=mk("Rail_Alu", (0.78, 0.79, 0.81), 0.35, 1.0),
             steel=mk("Steel", (0.6, 0.61, 0.63), 0.3, 1.0),
             black=mk("Black_Powdercoat", (0.03, 0.03, 0.035), 0.55),
             red=mk("Red", (0.85, 0.08, 0.06), 0.4),
             rubber=mk("Rubber", (0.06, 0.06, 0.06), 0.9),
             tyre=mk("Tyre", (0.35, 0.36, 0.38), 0.85),
             uhmw=mk("UHMW", (0.92, 0.92, 0.9), 0.6),
             bungee=mk("Bungee", (0.05, 0.05, 0.07), 0.7))

    root = empty("KITE_Catapult", None)
    th = math.radians(C["angle_deg"])
    rail = empty("Rail_Frame", root, (0, 0, C["pin_h"]), (0, -th, 0))
    Wm = Matrix.Translation(v(0, 0, C["pin_h"])) @ Matrix.Rotation(-th, 4, 'Y')

    sheave_z = build_rail(rail, M)
    build_rear(root, rail, M)
    build_hydraulic(root, M, {"rail": rail, "W": Wm})

    shuttle = empty("Shuttle", rail, (C["cock_x"], 0, TOP))
    x0, x1 = build_shuttle(shuttle, M)

    # bungee upper run: stretches automatically to the shuttle hooks
    for sgn in (1, -1):
        hook = empty(f"Bungee_Hook_Target_{'L' if sgn > 0 else 'R'}", shuttle, (x1 - 7, sgn * C["bungee_y"], 5))
        n = 12; r = C["bungee_d"] / 2
        vs = [(r * math.cos(2 * math.pi * i / n), y, r * math.sin(2 * math.pi * i / n))
              for y in (0, 1000) for i in range(n)]
        fs = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
        cord = mesh_obj(f"Bungee_Upper_{'L' if sgn > 0 else 'R'}", vs, fs, M["bungee"], rail, smooth=True)
        cord.location = v(C["sheave_x"], sgn * C["bungee_y"], TOP + 5)
        con = cord.constraints.new('STRETCH_TO')
        con.target = hook; con.rest_length = 1.0; con.volume = 'NO_VOLUME'

    # drone (child of rail frame, keyed so it separates at the buffers)
    droot = drone.build(clear=False)
    bpy.context.view_layer.update()
    droot.parent = rail
    droot.matrix_parent_inverse.identity()
    droot.location = v(C["cock_x"] + CANARD_TE, 0, TOP + C["cradle_h"])
    bpy.context.view_layer.update()
    sad = build_saddles(shuttle, M, [bpy.data.objects["Fuselage"], bpy.data.objects["Wing"]])

    # prop clearance visual (viewport only)
    bpy.ops.mesh.primitive_circle_add(vertices=64, radius=P["prop_d"] / 2 * MM, fill_type='NGON',
                                      location=v(-PROP_S, 0, PROP_Z), rotation=(0, math.pi / 2, 0))
    disc = bpy.context.object; disc.name = "Prop_Disc_Clearance"; disc.parent = droot
    disc.display_type = 'WIRE'; disc.hide_render = True

    animate(shuttle, droot, x1)
    report = dict(
        prop_tip_above_rail=C["cradle_h"] + PROP_Z - P["prop_d"] / 2,
        prop_plane_rail_x_cocked=C["cock_x"] + CANARD_TE - PROP_S,
        stroke=C["buffer_x"] - x1 - C["cock_x"] + 0,
        front_height=C["pin_h"] + C["rail_len"] * math.sin(th) + TOP * math.cos(th),
        saddles=sad)
    assert report["prop_tip_above_rail"] >= C["prop_clear_min"], "prop hits rail - raise cradle_h"
    return root, report

def animate(shuttle, droot, x1):
    sc = bpy.context.scene
    try:
        bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
    except Exception:
        pass
    fc, n, fe = C["f_cocked"], C["f_release_len"], C["f_end"]
    sc.frame_start, sc.frame_end, sc.render.fps = 1, fe, 24
    xc, xs = C["cock_x"], C["buffer_x"] - x1
    phi = 1.0
    def pos(u):
        return xc + (xs - xc) * (1 - math.cos(phi * u)) / (1 - math.cos(phi))
    vexit = (xs - xc) * phi * math.sin(phi) / (1 - math.cos(phi)) / n
    prop = bpy.data.objects["Propeller"]
    for f in range(1, fe + 1):
        if f <= fc:
            x, dz = xc, 0
        elif f <= fc + n:
            x, dz = pos((f - fc) / n), 0
        else:
            k = f - fc - n
            x, dz = xs, 0
        shuttle.location.x = x * MM
        shuttle.keyframe_insert("location", index=0, frame=f)
        if f <= fc + n:
            dx_d, dz_d = x, 0
        else:
            dx_d = xs + vexit * k
            dz_d = 0.4 * vexit * k * (k / 48)      # gentle climb-out
        droot.location.x = (dx_d + CANARD_TE) * MM
        droot.location.z = (TOP + C["cradle_h"] + dz_d) * MM
        droot.keyframe_insert("location", frame=f)
        prop.rotation_euler.x = math.radians(37 * f)
        prop.keyframe_insert("rotation_euler", index=0, frame=f)
    sc.frame_set(1)

if __name__ == "__main__":
    _, rep = build()
    for k, val in rep.items():
        print(k, ":", val)
