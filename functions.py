import torch.autograd as autograd
import torch
from core.pde import laplacian, grad
import numpy as np
import math


class ElectroThermalFunc(): 
 
 func_name = "Helmholtz_SingleInterface"
 def __init__(
     self,
     eps_inner: float = 15.0,
     eps_outer: float = 1.0,
     k_inner: float = 2.0,
     k_outer: float = 1.0,
     center: tuple = (0.5, 0.5),
     r1: float = 0.30,
     bc_tol: float = 1e-3,
 ):
     # store material properties and geometry
     self.eps_inner = eps_inner    # ε inside radius
     self.eps_outer = eps_outer    # ε outside
     self.k_inner   = k_inner      # wave‐number inside
     self.k_outer   = k_outer      # wave‐number outside
     self.cx, self.cy = center     # inclusion center
     self.r1 = r1                  # interface radius
     self.bc_tol = bc_tol          # tolerance for boundary masks
 
 def graph_modify(self, graph):
     """
     Attach node features [ x, y, ε(x), k(x) ] to graph.x.
     """
     # 1) raw coords
     x = graph.pos[:, 0:1]  # shape [N,1]
     y = graph.pos[:, 1:2]  # shape [N,1]
 
     # 2) radial distance from center
     dx = x - self.cx
     dy = y - self.cy
     r = torch.sqrt(dx * dx + dy * dy)  # [N,1]
 
     # 3) piecewise‐constant ε and k
     # if r ≤ r1 → inner, else → outer
     eps = torch.where(r <= self.r1,
                       torch.full_like(r, self.eps_inner),
                       torch.full_like(r, self.eps_outer))
     k   = torch.where(r <= self.r1,
                       torch.full_like(r, self.k_inner),
                       torch.full_like(r, self.k_outer))
 
     # 4) stack into graph.x
     graph.x = torch.cat([x, y, eps, k], dim=-1)  # [N,4]
     return graph
 
 def _ansatz_u(self, graph, u_raw):
    """
    Hard‐enforce u(0,y)=1, u(1,y)=0 via
       G(x,y)=1−x,   D(x,y)=x*(1−x),
    but detach both so they don't build extra graph nodes.
    """
    # pull out the x‐coordinate
    x = graph.pos[:, 0:1]    # shape [N,1]

    # compute G and D, then detach so no grads flow through them
    G = (1.0 - x).detach()           # guide function, exact on x=0→1, x=1→0
    D = (x * (1.0 - x)).detach()     # bump, zero at x=0 and x=1

    # our final ansatz: G + D * u_raw
    return G + D * u_raw

 
 def pde_residual(self, graph, u):
     """
     Compute r = ∇·(ε ∇u) + k^2 u at every node.
     Returns:
       r_pde  [N,1],   grad_u [N,2]
     """
     pos = graph.pos                       # [N,2]
     eps = graph.x[:, 2:3]                 # [N,1]
     k   = graph.x[:, 3:4]                 # [N,1]
 
     # 1) ∇u
     grad_u = torch.autograd.grad(
         outputs=u,
         inputs=pos,
         grad_outputs=torch.ones_like(u),
         create_graph=True
     )[0]                                 # [N,2]
 
     # 2) flux = ε ∇u
     flux = eps * grad_u                  # [N,2]
 
     # 3) div(flux)
     div = torch.zeros_like(u)
     for d in range(2):
         di = torch.autograd.grad(
             outputs=flux[:, d:d+1],
             inputs=pos,
             grad_outputs=torch.ones_like(flux[:, d:d+1]),
             create_graph=True
         )[0][:, d:d+1]
         div = div + di
 
     # 4) assemble residual (f≡0)
     r_pde = div + (k**2) * u
     return r_pde, grad_u
   

    
        

    
    

    
    
