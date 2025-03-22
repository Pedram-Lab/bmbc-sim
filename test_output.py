import os

import pyvista as pv
from netgen import occ
import ngsolve as ngs


if os.path.exists('test.vtu'):
    pv_mesh = pv.read('test_step00002.vtu')
    # pv_mesh['blub'] = pv_mesh['blub1'] + pv_mesh['blub2']
    print(f"arrays: {pv_mesh.array_names}")
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars='blub', show_edges=True)
    plotter.show()

else:
    cube1 = occ.Box((0, 0, 0), (1, 1, 1)).mat('left')
    cube2 = occ.Box((1, 0, 0), (2, 1, 1)).mat('right')
    cube3 = occ.Box((2, 0, 0), (3, 1, 1)).mat('cell')

    geo = occ.OCCGeometry(occ.Glue([cube1, cube2, cube3]))
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.2))

    # fes = ngs.H1(mesh, order=1, definedon='left|right')
    # u = ngs.GridFunction(fes)
    # u.Set(ngs.x * ngs.y)
    # ngs.VTKOutput(mesh, coefs=[u], names=['blub'], filename='test').Do()

    fes_left = ngs.Compress(ngs.H1(mesh, order=1, definedon='left|right'))
    fes_right = ngs.Compress(ngs.H1(mesh, order=1, definedon='cell'))
    fes = fes_left * fes_right

    u = ngs.GridFunction(fes)

    # print(f"fes_left dofs: {fes_left.ndof}")
    # print(f"fes_right dofs: {fes_right.ndof}")
    # print(f"fes dofs: {fes.ndof}")
    # print(f"fes_left grid function: {u_md.components[0].vec.FV().NumPy().shape}")
    # print(f"fes_right grid function: {u_md.components[1].vec.FV().NumPy().shape}")

    coeff = mesh.MaterialCF({
        'left': u.components[0],
        'right': u.components[0],
        'cell': u.components[1]
    })
    vtk = ngs.VTKOutput(mesh, coefs=[coeff], names=['blub'], filename='test')
    vtk.Do()
    vec_cf = ngs.CoefficientFunction((u.components[0], u.components[1]))

    u.components[0].Set(ngs.x * ngs.y)
    u.components[1].Set(1)
    mass_ecm_and_cell = ngs.Integrate(vec_cf, mesh, ngs.VOL)
    print(f"mass in ecs: {mass_ecm_and_cell[0]:.2f}")
    print(f"mass in cell: {mass_ecm_and_cell[1]:.2f}")
    vtk.Do()

    u.components[0].Set(0)
    u.components[1].Set(3.14)
    mass_ecm_and_cell = ngs.Integrate(vec_cf, mesh, ngs.VOL)
    print(f"mass in ecs: {mass_ecm_and_cell[0]:.2f}")
    print(f"mass in cell: {mass_ecm_and_cell[1]:.2f}")
    vtk.Do()
