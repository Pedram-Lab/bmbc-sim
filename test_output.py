import os

import pyvista as pv
from netgen import occ
import ngsolve as ngs


if os.path.exists('test.vtu'):
    pv_mesh = pv.read('test_step00002.vtu')
    pv_mesh['blub'] = pv_mesh['blub1'] + pv_mesh['blub2']
    print(f"arrays: {pv_mesh.array_names}")
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars='blub', show_edges=True)
    plotter.show()

else:
    cube1 = occ.Box((0, 0, 0), (1, 1, 1)).mat('left')
    cube2 = occ.Box((1, 0, 0), (2, 1, 1)).mat('right')

    geo = occ.OCCGeometry(occ.Glue([cube1, cube2]))
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.1))

    # fes = ngs.H1(mesh, order=1, definedon='left|right')
    # u = ngs.GridFunction(fes)
    # u.Set(ngs.x * ngs.y)
    # ngs.VTKOutput(mesh, coefs=[u], names=['blub'], filename='test').Do()

    fes_left = ngs.Compress(ngs.H1(mesh, order=1, definedon='left'))
    fes_right = ngs.Compress(ngs.H1(mesh, order=1, definedon='right'))
    fes = fes_left * fes_right

    u = ngs.GridFunction(fes)

    # print(f"fes_left dofs: {fes_left.ndof}")
    # print(f"fes_right dofs: {fes_right.ndof}")
    # print(f"fes dofs: {fes.ndof}")
    # print(f"fes_left grid function: {u_md.components[0].vec.FV().NumPy().shape}")
    # print(f"fes_right grid function: {u_md.components[1].vec.FV().NumPy().shape}")

    vtk = ngs.VTKOutput(mesh, coefs=[u.components[0], u.components[1]], names=['blub1', 'blub2'], filename='test')
    vtk.Do()

    u.components[0].Set(ngs.x * ngs.y)
    u.components[1].Set(1)
    vtk.Do()

    u.components[0].Set(ngs.x * ngs.z)
    u.components[1].Set(2)
    vtk.Do()
