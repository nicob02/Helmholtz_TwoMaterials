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

from fenics import (
    Point, FunctionSpace, TrialFunction, TestFunction, Function,
    Constant, DirichletBC, solve, dx, dot, grad, conditional, le,
    interpolate, SpatialCoordinate
)

from fenics import (
    Point, FunctionSpace, TrialFunction, TestFunction,
    Function, Constant, DirichletBC, solve, dx,
    dot, grad, conditional, le, SpatialCoordinate
)
import numpy as np

def run_fem(
    electrode_mesh,
    coords=None,
    r1=0.30,
    eps_inner=20.0, eps_outer=1.0,
    k_inner=0.2, k_outer=1.0
):
    """
    Solve ∇·(ε ∇u) + k^2 u = 0 on [0,1]^2 with one inclusion.
      • Dirichlet u=1 on x=0,
        Dirichlet u=0 on x=1,
        natural (Neumann) on y=0,1.
    Returns (coords, Uvals).
    """
    mesh = electrode_mesh.mesh
    V    = FunctionSpace(mesh, 'P', 1)

    u = TrialFunction(V)
    v = TestFunction(V)
    X = SpatialCoordinate(mesh)

    # piecewise ε and k
    eps = conditional(
        le((X[0]-0.5)**2 + (X[1]-0.5)**2, r1**2),
        Constant(eps_inner),
        Constant(eps_outer)
    )
    kk = conditional(
        le((X[0]-0.5)**2 + (X[1]-0.5)**2, r1**2),
        Constant(k_inner),
        Constant(k_outer)
    )

    # weak form: ∫ eps ∇u·∇v dx - ∫ k^2 u v dx = 0
    a = dot(eps*grad(u), grad(v))*dx - kk**2 * u*v*dx
    L = Constant(0.0) * v * dx

    # Dirichlet BC on x=0: u=1
    def left_boundary(x, on_bdry):
        return on_bdry and abs(x[0]) < 1e-8
    bc_left = DirichletBC(V, Constant(1.0), left_boundary)

    # Dirichlet BC on x=1: u=0
    def right_boundary(x, on_bdry):
        return on_bdry and abs(x[0] - 1.0) < 1e-8
    bc_right = DirichletBC(V, Constant(0.0), right_boundary)

    bcs = [bc_left, bc_right]

    U = Function(V)
    solve(a == L, U, bcs)

    # sample at graph.pos (or provided coords)
    if coords is None:
        coords = electrode_mesh.pos  # numpy array of shape [N,2]

    Uvals = np.zeros((coords.shape[0],), dtype=float)
    for i, (xi, yi) in enumerate(coords):
        Uvals[i] = U(Point(float(xi), float(yi)))

    return coords, Uvals
