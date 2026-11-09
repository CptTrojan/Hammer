import bpy
import bmesh
import math

# ── Clear scene ──────────────────────────────────────────
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# ── Dimensions (in meters) ───────────────────────────────
WINGSPAN        = 0.900
FUSELAGE_LEN    = 0.700
ROOT_CHORD      = 0.400
TIP_CHORD       = 0.150
NOSE_LEN        = 0.100
CANARD_CHORD    = 0.100
FUSELAGE_W      = 0.090   # fuselage width estimate
FUSELAGE_H      = 0.080   # fuselage height estimate
CG_FROM_NOSE    = 0.370   # 13 inches from tail

# ── Helper: create mesh object ───────────────────────────
def make_obj(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    obj  = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    bv = [bm.verts.new(v) for v in verts]
    bm.verts.ensure_lookup_table()
    for f in faces:
        bm.faces.new([bv[i] for i in f])
    bm.to_mesh(mesh)
    bm.free()
    return obj

# ── Fuselage (box) ───────────────────────────────────────
hw = FUSELAGE_W / 2
hh = FUSELAGE_H / 2
fl = FUSELAGE_LEN
fuse_verts = [
    (0,   -hw, -hh), (0,   hw, -hh), (0,   hw,  hh), (0,  -hw,  hh),
    (fl,  -hw, -hh), (fl,  hw, -hh), (fl,  hw,  hh), (fl, -hw,  hh),
]
fuse_faces = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(0,3,7,4),(1,2,6,5)]
fuse = make_obj("Fuselage", fuse_verts, fuse_faces)

# ── Wing (tapered, both sides) ───────────────────────────
# Wing leading edge starts at 30% fuselage from nose
wing_le_x = FUSELAGE_LEN * 0.30
half_span  = (WINGSPAN - FUSELAGE_W) / 2

# Left wing
wl_verts = [
    (wing_le_x,               -hw,        0.002),
    (wing_le_x + ROOT_CHORD,  -hw,        0.002),
    (wing_le_x + ROOT_CHORD + (TIP_CHORD - ROOT_CHORD)*0.3,
                              -(hw + half_span), 0.002),
    (wing_le_x + ROOT_CHORD - TIP_CHORD + (TIP_CHORD - ROOT_CHORD)*0.3,
                              -(hw + half_span), 0.002),
]
# Right wing (mirror)
wr_verts = [(x, -y, z) for (x, y, z) in wl_verts]

wl = make_obj("Wing_L", wl_verts, [(0,1,2,3)])
wr = make_obj("Wing_R", wr_verts, [(0,1,2,3)])

# ── Canard ───────────────────────────────────────────────
canard_x   = NOSE_LEN + 0.020
canard_span = 0.200
cn_verts = [
    (canard_x,                  -canard_span/2, 0.002),
    (canard_x + CANARD_CHORD,   -canard_span/2, 0.002),
    (canard_x + CANARD_CHORD,    canard_span/2, 0.002),
    (canard_x,                   canard_span/2, 0.002),
]
canard = make_obj("Canard", cn_verts, [(0,1,2,3)])

# ── CG Marker ────────────────────────────────────────────
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.015,
    location=(CG_FROM_NOSE, 0, 0))
cg = bpy.context.active_object
cg.name = "CG_Marker"
mat = bpy.data.materials.new("CG_Red")
mat.diffuse_color = (1, 0, 0, 1)
cg.data.materials.append(mat)

# ── Nose cone (simple cone) ──────────────────────────────
bpy.ops.mesh.primitive_cone_add(radius1=hw, depth=NOSE_LEN,
    location=(NOSE_LEN/2, 0, 0),
    rotation=(0, math.pi/2, 0))
nose = bpy.context.active_object
nose.name = "Nose"

print("✓ Aircraft model built successfully")
print(f"  CG at {CG_FROM_NOSE*1000:.0f}mm from nose (red sphere)")
