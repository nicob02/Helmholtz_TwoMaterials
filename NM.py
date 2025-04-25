from petsc4py import PETSc
from fenics import (NonlinearProblem, NewtonSolver, FiniteElement,
                    MixedElement, assemble, FunctionSpace, TestFunctions, Function,
                    interpolate, Expression, split, dot, inner, grad, dx, DirichletBC,
                    Constant, exp, ln, derivative, PETScKrylovSolver,
                    PETScFactory, near, PETScOptions, assign, File, plot, SpatialCoordinate)
from ufl import conditional
import numpy as np
import sys
import matplotlib.pyplot as plt
from core.geometry import ElectrodeMesh
from fenics import (
    Point, FunctionSpace, TrialFunction, TestFunction, Function,
    Constant, DirichletBC, solve, dx, dot, grad
)
from ufl import conditional, le, SpatialCoordinate

def run_fem(electrode_mesh, coords=None,
            r1=0.15, r2=0.30,
            eps1=4.0, eps2=2.0, eps3=1.0,
            k1=20.0, k2=10.0, k3=5.0):
    """
    Solve ∇·(ε(x)∇u) + k(x)^2 u = 0 on the unit square with two concentric
    circular inclusions of radius r1,r2, piecewise-constant ε,k.
    Dirichlet u=1 on x=0; Neumann (natural) elsewhere.

    Returns coords and the FEM solution sampled at those coords.
    """
    # 1) pull out the underlying dolfin mesh
    mesh = electrode_mesh.mesh

    # 2) scalar P1 space
    V_space = FunctionSpace(mesh, 'CG', 1)

    # 3) define trial/test
    u = TrialFunction(V_space)
    v = TestFunction(V_space)

    # 4) build piecewise ε,k via SpatialCoordinate
    X   = SpatialCoordinate(mesh)
    eps = conditional(
        le((X[0]-0.5)**2 + (X[1]-0.5)**2, r1**2),
        Constant(eps1),
        conditional(
            le((X[0]-0.5)**2 + (X[1]-0.5)**2, r2**2),
            Constant(eps2),
            Constant(eps3)
        )
    )
    kk = conditional(
        le((X[0]-0.5)**2 + (X[1]-0.5)**2, r1**2),
        Constant(k1),
        conditional(
            le((X[0]-0.5)**2 + (X[1]-0.5)**2, r2**2),
            Constant(k2),
            Constant(k3)
        )
    )

    # 5) weak form: ∫ ε ∇u·∇v dx  - ∫ k^2 u v dx = 0
    a = dot(eps*grad(u), grad(v))*dx - kk**2 * u*v*dx
    L = Constant(0.0)*v*dx

    # 6) Dirichlet on x=0: u=1
    def left_boundary(x_, on_bnd):
        return on_bnd and abs(x_[0]) < 1e-8
    bc = DirichletBC(V_space, Constant(1.0), left_boundary)

    # 7) solve
    U = Function(V_space)
    solve(a == L, U, bc)

    # 8) sample at the same coords as graph.pos
    if coords is None:
        coords = electrode_mesh.pos

    VV = np.empty((coords.shape[0],), float)
    for i, (xi, yi) in enumerate(coords):
        p = Point(float(xi), float(yi))
        VV[i] = U(p)

    return coords, VV
