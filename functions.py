import torch.autograd as autograd
import torch
from core.pde import laplacian, grad
import numpy as np
import math


class ElectroThermalFunc(): 

    func_name = 'Helmholtz_Left'
    def __init__(self,
                 eps=(4.0, 2.0, 1.0),
                 k  =(20.0,10.0, 5.0),
                 center=(0.5,0.5),
                 r1=0.15,
                 r2=0.30,
                 bc_tol=1e-3):
        self.eps1,self.eps2,self.eps3 = eps
        self.k1,  self.k2,  self.k3   = k
        self.cx,  self.cy            = center
        self.r1,  self.r2            = r1, r2
        self.bc_tol                  = bc_tol

    def graph_modify(self, graph):
        """
        Build node features [x, y, eps(x), k(x)].
        """
        x = graph.pos[:,0:1]
        y = graph.pos[:,1:2]
        dx = x - self.cx
        dy = y - self.cy
        r  = torch.sqrt(dx*dx + dy*dy)

        eps = torch.where(r <= self.r1,
                          self.eps1,
                   torch.where(r <= self.r2,
                               self.eps2,
                               self.eps3))
        k   = torch.where(r <= self.r1,
                          self.k1,
                   torch.where(r <= self.r2,
                               self.k2,
                               self.k3))

        graph.x = torch.cat([x, y, eps, k], dim=-1)   # all are [N,1]
        return graph

    def _ansatz_u(self, graph, u_raw):
        """
        Hard Dirichlet on x=0 (“incoming plane wave” u=cos(k3 x))
        G(x)=cos(k3 x), D(x)=tanh(pi x) so that u(0)=G(0)=1.
        """
        x = graph.pos[:, 0:1]
        # detach so no grad‐graph is built back through 'x'
        G = torch.cos(self.k3 * x).detach()  
        D = torch.tanh(math.pi * x).detach()
    
        # Only the multiplication by u_raw remains in the graph.
        return G + D * u_raw

    def pde_residual(self, graph, u):
        """
        Computes (div(eps * grad u) + k^2 u) at every node.
        Returns
          r_pde  [N,1],  grad_u [N,2]
        """
        pos = graph.pos
        eps = graph.x[:,2:3]
        k   = graph.x[:,3:4]

        # ∇u
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=pos,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0]                                         # [N,2]

        # flux = eps * ∇u
        flux = eps * grad_u                          # [N,2]

        # div flux
        div = torch.zeros_like(u)
        for i in range(2):
            di = torch.autograd.grad(
                outputs=flux[:,i:i+1],
                inputs=pos,
                grad_outputs=torch.ones_like(flux[:,i:i+1]),
                create_graph=True,
            )[0][:,i:i+1]
            div = div + di

        # PDE residual = div(eps grad u) + k^2 u   (f=0)
        r_pde = div + (k**2)*u
        return r_pde, grad_u

    
    

    
        

    
    

    
    
