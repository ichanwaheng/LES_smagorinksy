#!/usr/bin/env python
# coding: utf-8

# In[12]:


import gmsh
import sys


# In[ ]:


def generate_3d_fluid_mesh():
    gmsh.initialize()
    try:
        gmsh.model.add("OpenFOAM_3D_Unstructured_mesh")

        # Geometry parameters (fluid domain)
        L, W, H = 10, 5, 5  # (in metres)
        r = 0.5  # radius of sphere in the fluid domain
        cx, cy, cz = 3, 2.5, 2.5  # centre position of the sphere

        # Create geometry in gmsh
        box = gmsh.model.occ.addBox(0, 0, 0, L, W, H)
        sphere = gmsh.model.occ.addSphere(cx, cy, cz, r)

        # Fluid domain: box - sphere
        fluid_v, _ = gmsh.model.occ.cut([(3, box)], [(3, sphere)])
        gmsh.model.occ.synchronize()

        # Identify boundaries
        surfaces = gmsh.model.getBoundary(fluid_v, oriented=False)
        inlet, outlet, walls, object_surf = [], [], [], []

        for dim, tag in surfaces:
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            if abs(com[0] - 0) < 1e-3:
                inlet.append(tag)
            elif abs(com[0] - L) < 1e-3:
                outlet.append(tag)
            elif (cx - r - 0.1 < com[0] < cx + r + 0.1) and (cy - r - 0.1 < com[1] < cy + r + 0.1):
                object_surf.append(tag)
            else:
                walls.append(tag)

        # Physical groups
        gmsh.model.addPhysicalGroup(2, inlet, name="inlet")
        gmsh.model.addPhysicalGroup(2, outlet, name="outlet")
        gmsh.model.addPhysicalGroup(2, walls, name="walls")
        gmsh.model.addPhysicalGroup(2, object_surf, name="object")

        fluid_tags = [tag for dim, tag in fluid_v]
        gmsh.model.addPhysicalGroup(3, fluid_tags, name="internalfield")

        # Refinement fields
        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "SurfacesList", object_surf)

        f_thresh = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", 0.05)
        gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", 1)
        gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.2)
        gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", 2.5)
        gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)

        # Mesh options
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

        # Generate mesh
        gmsh.model.mesh.generate(3)
        gmsh.write("fluid_mesh_3d.msh")

        # Visualization settings (best-effort; options differ across versions)
        options = [
            ("Mesh.SurfaceFaces", 1),
            ("Mesh.Volume", 0),
            ("Mesh.Volumes", 0),
            ("Geometry.SurfaceAlpha", 40),
            ("General.AlphaBlending", 1),
        ]

        for opt, val in options:
            try:
                gmsh.option.setNumber(opt, val)
            except Exception:
                print(f"[NOTE] Option {opt} not supported in this version. Skipping.")

        if "-nopopup" not in sys.argv:
            print("\n[SUCCESS] GUI Opening.")
            print("To see inside: Press 'a' on your keyboard to toggle transparency.")
            gmsh.fltk.run()
    finally:
        gmsh.finalize()



if __name__ == "__main__":
    generate_3d_fluid_mesh()


# In[ ]:




