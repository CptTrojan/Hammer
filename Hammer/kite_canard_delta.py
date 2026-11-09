"""
KITE canard-delta pusher UAV  -  parametric Blender model
=========================================================
Run inside Blender:  Scripting tab -> Open this file -> Run Script
or headless:         blender --background --python kite_canard_delta.py

All dimensions are in millimetres, measured from the reference photos
(floor tiles = 1 ft = 304.8 mm, ~0.685 mm/px).  Edit the PARAMETERS block
and re-run to regenerate.  Axes: +X = forward (nose), +Y = left wing, +Z = up.
Origin = nose tip.  "Station" (s) = distance aft of the nose tip.
"""
import bpy, bmesh, math
from mathutils import Vector

# ----------------------------------------------------------------- PARAMETERS
P = dict(
    # fuselage
    fus_radius=31.0,          # Ø62 tube
    nose_len=100.0,           # 100 mm nose cone
    tube_end=290.0,           # station where the tube blends into the wing body
    body_end=700.0,           # station of trailing edge (fuselage length = 700 mm)
    # main wing (cranked straight TE, swept LE)
    wing_z=-6.0,              # chord-plane height relative to fuselage axis
    root_le=300.0,            # LE station at centreline: te(700) - root_chord(400) = 300
    tip_le=550.0,             # LE station at the tip:    te(700) - tip_chord(150)  = 550
    te=700.0,                 # straight trailing edge station (= fuselage length)
    semispan=450.0,           # 900 mm wingspan / 2 = 450 mm semispan
    wing_t_root=0.11,         # thickness/chord at root
    wing_t_tip=0.09,          # thickness/chord at tip
    wing_camber=0.0,          # NACA 4-digit max camber (0 = symmetric)
    # elevons (black)
    elev_chord=75.0,          # scaled proportionally
    elev_y_in=140.0,          # inner edge of elevon bay (from centreline)
    elev_y_out=380.0,         # outer edge of elevon bay (from centreline)
    elev_gap=1.5,
    # canards  — root = tip = 100 mm (rectangular planform)
    can_le=105.0,             # canard LE station (just after nose)
    can_root_chord=100.0,     # 100 mm
    can_tip_chord=100.0,      # 100 mm (rectangular)
    can_semispan=193.0,       # 386 mm tip-to-tip (user spec not given, kept)
    can_t=0.12,
    can_z=0.0,
    # wing-tip fins (drooped plates) — scaled to new body_end
    fin_s_top_le=675.0, fin_s_top_te=744.0,
    fin_s_bot_le=710.0, fin_s_bot_te=744.0,
    fin_height=28.0, fin_thick=2.5, fin_cant_deg=10.0,
    # pusher power train  — A2212 motor + 1045 prop
    pylon_s0=540.0, pylon_s1=680.0, pylon_h=24.0, pylon_t=8.0,
    motor_d=28.0, motor_len=26.0, motor_axis_z=18.0,
    prop_d=254.0,             # 1045 prop = 10 in = 254 mm
    prop_chord=18.0,
    # CG — nose(100mm) + 330mm from tail ref = 430 mm from nose tip
    cg_from_nose=430.0,
)
MM = 0.001   # model in metres, displayed in mm

# ------------------------------------------------------------------ helpers
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.materials):
        for b in list(block):
            block.remove(b)

def mat(name, rgb, rough=0.5, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    m.diffuse_color = (*rgb, 1)
    return m

def W(s, y, z):
    """station/span/height in mm -> world metres (nose at +X)."""
    return Vector((-s * MM, y * MM, z * MM))

def make_obj(name, verts, faces, material=None, parent=None, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.validate()
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    if material:
        ob.data.materials.append(material)
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
        ob.modifiers.new("WN", 'WEIGHTED_NORMAL').keep_sharp = True
    if parent:
        ob.parent = parent
    return ob

def smooth_angle(ob, deg=35):
    for o in bpy.context.selected_objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(deg))

def loft(sections, cap_start=True, cap_end=True):
    """sections: list of closed rings, each a list of Vector (same count)."""
    n = len(sections[0])
    verts = [v for ring in sections for v in ring]
    faces = []
    for i in range(len(sections) - 1):
        a, b = i * n, (i + 1) * n
        for j in range(n):
            k = (j + 1) % n
            faces.append((a + j, a + k, b + k, b + j))
    if cap_start:
        faces.append(tuple(reversed(range(n))))
    if cap_end:
        last = (len(sections) - 1) * n
        faces.append(tuple(range(last, last + n)))
    return verts, faces

def naca(x, t, m=0.0, p=0.4):
    yt = 5 * t * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 - 0.1036 * x**4)
    if m == 0:
        return yt, -yt
    yc = m / p**2 * (2*p*x - x*x) if x < p else m / (1-p)**2 * ((1-2*p) + 2*p*x - x*x)
    return yc + yt, yc - yt

def airfoil_ring(x0, x1, npts, t, m=0.0):
    """Closed (x,z) ring of a NACA section clipped to chord fraction [x0,x1]."""
    xs = [x0 + (x1 - x0) * 0.5 * (1 - math.cos(math.pi * i / (npts - 1)))
          for i in range(npts)]
    up = [(x, naca(x, t, m)[0]) for x in reversed(xs)]
    lo = [(x, naca(x, t, m)[1]) for x in xs]
    if x0 == 0.0:
        lo = lo[1:]                  # shared LE point
    return up + lo

def wing_section(y, le, chord, z, t, m, x0=0.0, x1=1.0, npts=28):
    return [W(le + chord * x, y, z + chord * zz) for x, zz in airfoil_ring(x0, x1, npts, t, m)]

def mirror_y(verts, faces):
    n = len(verts)
    v2 = [Vector((v.x, -v.y, v.z)) for v in verts]
    f2 = [tuple(reversed([i + n for i in f])) for f in faces]
    return verts + v2, faces + f2

# --------------------------------------------------------------- components
def build_fuselage(root, m_white):
    R, Ln = P["fus_radius"], P["nose_len"]
    N = 40
    def ring(s, hw, ht, hb, zc=0.0, sq=2.2):
        pts = []
        for i in range(N):
            a = 2 * math.pi * i / N
            c, sn = math.cos(a), math.sin(a)
            y = hw * math.copysign(abs(c) ** (2 / sq), c)
            h = ht if sn >= 0 else hb
            z = zc + h * math.copysign(abs(sn) ** (2 / sq), sn)
            pts.append(W(s, y, z))
        return pts

    # nose cone + tube + blended centre body — single continuous loft (no seam)
    rho = (R**2 + Ln**2) / (2 * R)
    all_secs = []

    # nose cone: tangent ogive
    for i in range(13):
        s = max(0.4, Ln * (1 - math.cos(math.pi / 2 * i / 12)))
        r = math.sqrt(rho**2 - (Ln - s) ** 2) + R - rho
        all_secs.append(ring(s, r, r, r, sq=2.0))

    # tube section (skip first ring — shared with nose end)
    s0, s1, s2 = Ln, P["tube_end"], P["body_end"]
    for i in range(1, 10):
        s = s0 + (s1 - s0) * i / 9
        all_secs.append(ring(s, R, R, R, sq=2.0))

    # blended centre body
    for i in range(1, 21):
        u = i / 20
        s = s1 + (s2 - 4 - s1) * u
        e = 0.5 - 0.5 * math.cos(math.pi * u)            # smooth 0->1
        hw = R + 4 * math.sin(math.pi * min(1, u * 1.4)) - (R - 15) * e ** 1.5
        ht = R - (R - 7) * e
        hb = R - (R - 11) * e
        zc = P["wing_z"] * e
        all_secs.append(ring(s, hw, ht, hb, zc, sq=2.0 + 1.2 * e))

    v, f = loft(all_secs)
    make_obj("Fuselage", v, f, m_white, root)

def build_wing(root, m_white, m_black):
    p = P
    def le_at(y):
        return p["root_le"] + (p["tip_le"] - p["root_le"]) * y / p["semispan"]
    def chord_at(y):
        return p["te"] - le_at(y)
    def t_at(y):
        return p["wing_t_root"] + (p["wing_t_tip"] - p["wing_t_root"]) * y / p["semispan"]
    z, m = p["wing_z"], p["wing_camber"]
    hinge_s = p["te"] - p["elev_chord"]

    def seg(y0, y1, nsec, clip):
        secs = []
        for i in range(nsec):
            y = y0 + (y1 - y0) * i / (nsec - 1)
            c, le = chord_at(y), le_at(y)
            x1 = (hinge_s - le) / c if clip else 1.0
            secs.append(wing_section(y, le, c, z, t_at(y), m, 0.0, x1))
        return secs

    # wing: one smooth loft, elevon bays cut out with booleans
    ys = [p["semispan"] * (1 - math.cos(math.pi / 2 * i / 15)) for i in range(16)]
    ys = sorted(set(ys + [p["elev_y_in"], p["elev_y_out"]]))
    secs = [wing_section(y, le_at(y), chord_at(y), z, t_at(y), m, npts=36) for y in ys]
    secs = [[Vector((q.x, -q.y, q.z)) for q in r] for r in secs[:0:-1]] + secs
    v, f = loft(secs)
    wing = make_obj("Wing", v, f, m_white, root, smooth=False)
    for sgn in (1, -1):
        g = p["elev_gap"]
        L = p["elev_chord"] + g + 10
        cy = sgn * (p["elev_y_in"] + p["elev_y_out"]) / 2
        bpy.ops.mesh.primitive_cube_add(size=1, location=W(p["te"] - L / 2 + 10 + g, cy, z))
        cut = bpy.context.object
        cut.scale = (L * MM, (p["elev_y_out"] - p["elev_y_in"]) * MM, 80 * MM)
        bo = wing.modifiers.new("cut", 'BOOLEAN'); bo.object = cut; bo.operation = 'DIFFERENCE'
        bpy.context.view_layer.objects.active = wing
        bpy.ops.object.modifier_apply(modifier=bo.name)
        bpy.data.objects.remove(cut)
    smooth_angle(wing)

    # elevons, origin on hinge line so they can be rotated about local Y
    for side, sgn in (("L", 1), ("R", -1)):
        secs = []
        g = p["elev_gap"]
        ya, yb = p["elev_y_in"] + g, p["elev_y_out"] - g
        for i in range(6):
            y = ya + (yb - ya) * i / 5
            c, le = chord_at(y), le_at(y)
            x0 = (hinge_s + g - le) / c
            ring = wing_section(y, le, c, z, t_at(y), m, x0, 1.0, npts=10)
            secs.append([Vector((q.x, q.y * sgn, q.z)) for q in ring])
        if sgn < 0:
            secs = secs[::-1]
        v, f = loft(secs)
        pivot = W(hinge_s + g, sgn * (ya + yb) / 2, z)
        v = [q - pivot for q in v]
        ob = make_obj(f"Elevon_{side}", v, f, m_black, root)
        ob.location = pivot

    # wing-tip fins (drooped plates)
    tip_y = p["semispan"]
    for side, sgn in (("L", 1), ("R", -1)):
        h, th = p["fin_height"], p["fin_thick"] / 2
        prof = [(p["fin_s_top_le"], 0), (p["fin_s_top_te"], 0),
                (p["fin_s_bot_te"], -h), (p["fin_s_bot_le"], -h)]
        cant = math.radians(p["fin_cant_deg"]) * sgn
        verts = []
        for dy in (-th, th):
            for s, zz in prof:
                yy = tip_y * sgn + dy + (-zz) * math.sin(cant)
                verts.append(W(s, yy, z + zz * math.cos(cant)))
        faces = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2),
                 (2, 6, 7, 3), (3, 7, 4, 0)]
        make_obj(f"TipFin_{side}", verts, faces, m_white, root, smooth=False)

def build_canards(root, m_white):
    p = P
    secs = []
    ys = [-p["can_semispan"], -p["can_semispan"] + 4, 0, p["can_semispan"] - 4, p["can_semispan"]]
    for y in ys:
        f = abs(y) / p["can_semispan"]
        c = p["can_root_chord"] + (p["can_tip_chord"] - p["can_root_chord"]) * f
        if abs(y) == p["can_semispan"]:      # small tip roll-off
            le, c = p["can_le"] + 3, c - 6
            t = p["can_t"] * 0.6
        else:
            le, t = p["can_le"], p["can_t"]
        secs.append(wing_section(y, le, c, p["can_z"], t, 0.0))
    v, f = loft(secs)
    make_obj("Canards", v, f, m_white, root)

def build_powertrain(root, m_white, m_motor, m_black, m_metal):
    p = P
    # tapered pylon on the spine
    top_z = p["wing_z"] + 14
    prof = []
    n = 14
    for i in range(n):
        u = i / (n - 1)
        s = p["pylon_s0"] + (p["pylon_s1"] - p["pylon_s0"]) * u
        prof.append((s, top_z - 6 + (p["pylon_h"] + 6) * u ** 1.3))
    verts = []
    for s, zt in prof:
        verts += [W(s, 0, top_z - 8), W(s, p["pylon_t"] / 2, (zt + top_z - 8) / 2),
                  W(s, 0, zt), W(s, -p["pylon_t"] / 2, (zt + top_z - 8) / 2)]
    v, f = loft([verts[i:i + 4] for i in range(0, len(verts), 4)])
    make_obj("Motor_Pylon", v, f, m_white, root)

    # motor (bell faces aft)
    mz = p["motor_axis_z"] + p["wing_z"]
    s_mount = p["pylon_s1"]
    bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=p["motor_d"] / 2 * MM,
                                        depth=p["motor_len"] * MM,
                                        location=W(s_mount + p["motor_len"] / 2, 0, mz),
                                        rotation=(0, math.pi / 2, 0))
    motor = bpy.context.object; motor.name = "Motor"; motor.parent = root
    motor.data.materials.append(m_motor)
    smooth_angle(motor)

    s_prop = s_mount + p["motor_len"] + 10
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=2.5 * MM, depth=22 * MM,
                                        location=W(s_prop - 4, 0, mz), rotation=(0, math.pi / 2, 0))
    shaft = bpy.context.object; shaft.name = "Prop_Shaft"; shaft.parent = root
    shaft.data.materials.append(m_metal)

    # 2-blade prop with twist; origin on the shaft so it can spin about X
    R = p["prop_d"] / 2
    verts, faces = [], []
    for sgn in (1, -1):
        base = len(verts)
        stations = 10
        for i in range(stations):
            r = 8 + (R - 8) * i / (stations - 1)
            c = p["prop_chord"] * (0.75 + 0.45 * math.sin(math.pi * min(1, r / R * 1.15)))
            if i == stations - 1:
                c *= 0.5
            pitch = math.atan2(229 * 0.6, 2 * math.pi * r)    # ~9x5.5-ish twist
            th = 2.2 * (1 - 0.5 * r / R)
            ring = [(-c / 2, 0), (c / 2, th * 0.3), (c / 2, -th * 0.2), (-c / 2, -th * 0.5)]
            for cx, cz in ring:
                ty = cx * math.cos(pitch) * sgn
                tx = cx * math.sin(pitch) + cz * math.cos(pitch)
                verts.append(Vector((-tx * MM, ty * MM, sgn * r * MM)))
        for i in range(stations - 1):
            a, b = base + i * 4, base + (i + 1) * 4
            for j in range(4):
                k = (j + 1) % 4
                faces.append((a + j, a + k, b + k, b + j) if sgn > 0 else (a + j, b + j, b + k, a + k))
        faces.append(tuple(range(base, base + 4))[::sgn])
        last = base + (stations - 1) * 4
        faces.append(tuple(range(last, last + 4))[::-sgn])
    prop = make_obj("Propeller", verts, faces, m_black, root, smooth=False)
    prop.location = W(s_prop, 0, mz)
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=6 * MM, depth=8 * MM,
                                        location=W(s_prop, 0, mz), rotation=(0, math.pi / 2, 0))
    hub = bpy.context.object; hub.name = "Prop_Hub"; hub.parent = root
    hub.data.materials.append(m_metal)

# ---------------------------------------------------------------------- main
def build(clear=True):
    if clear:
        clear_scene()
    sc = bpy.context.scene
    sc.unit_settings.system = 'METRIC'
    sc.unit_settings.length_unit = 'MILLIMETERS'

    m_white = mat("PLA_White", (0.85, 0.85, 0.82), 0.45)
    m_black = mat("Elevon_Black", (0.03, 0.03, 0.03), 0.8)
    m_motor = mat("Motor_Orange", (0.8, 0.35, 0.05), 0.35, 0.6)
    m_metal = mat("Aluminium", (0.8, 0.8, 0.82), 0.25, 1.0)

    root = bpy.data.objects.new("KITE_Canard_Delta", None)
    root.empty_display_type = 'ARROWS'
    root.empty_display_size = 0.1
    sc.collection.objects.link(root)

    build_fuselage(root, m_white)
    build_wing(root, m_white, m_black)
    build_canards(root, m_white)
    build_powertrain(root, m_white, m_motor, m_black, m_metal)

    # CG marker — red sphere at station 370 mm from nose
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=14 * MM,
        location=W(P["cg_from_nose"], 0, 0)
    )
    cg_obj = bpy.context.active_object
    cg_obj.name = "CG_Marker"
    cg_obj.parent = root
    m_cg = mat("CG_Red", (1.0, 0.05, 0.05), 0.2)
    cg_obj.data.materials.append(m_cg)

    return root

if __name__ == "__main__":
    build()
    print("KITE canard-delta built.")
    print("  Wingspan     : %.0f mm" % (2 * P["semispan"]))
    print("  Fuselage len : %.0f mm" % P["body_end"])
    print("  Root chord   : %.0f mm" % (P["te"] - P["root_le"]))
    print("  Tip chord    : %.0f mm" % (P["te"] - P["tip_le"]))
    print("  Nose length  : %.0f mm" % P["nose_len"])
    print("  Canard chord : %.0f mm (root=tip)" % P["can_root_chord"])
    print("  Prop         : 1045 (%.0f mm dia)" % P["prop_d"])
    print("  CG           : %.0f mm from nose tip  (red sphere)" % P["cg_from_nose"])
