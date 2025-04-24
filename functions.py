
import torch
from core.pde import laplacian, grad
import numpy as np
import math

class MagneticFunc:
    func_name = "magnetic_disc"

    def __init__(self,
                 mu_in: float = 3.0,
                 mu_out: float = 1.0,
                 center=(0.5,0.5),
                 radius=0.2,
                 steep=10.0):
        self.mu_in, self.mu_out = mu_in, mu_out
        self.cx, self.cy       = center
        self.r0                = radius
        self.steep             = steep

    def graph_modify(self, graph):
        """
        Append permeability mu as a node feature.
        graph.pos: [N,2]
        We build mu = mu_out + (mu_in-mu_out) * sigmoid((r0 - r)*steep)
        """
        x = graph.pos[:,0:1]
        y = graph.pos[:,1:2]
        # compute distance to center
        r = torch.sqrt((x-self.cx)**2 + (y-self.cy)**2)
        h = torch.sigmoid((self.r0 - r)*self.steep)  # approx Heaviside(r<r0)
        mu = self.mu_out + (self.mu_in-self.mu_out)*h    # [N,1]
        graph.x = torch.cat([x, y, mu], dim=-1)  # [N,3]
        return graph

    def _ansatz_Hz(self, graph, Hz_raw):
        """
        Hard‐enforce Hz=1 on x=0 and x=1 faces via:
         Hz = G + D*Hz_raw,
        where G≡1, D(x)=tanh(pi*x)*tanh(pi*(1-x))
        """
        x = graph.pos[:,0:1]
        D = torch.tanh(torch.pi*x)*torch.tanh(torch.pi*(1.0-x))
        G = torch.ones_like(x)  # Hz=1 on left/right
        return G + D*Hz_raw

    def pde_residual(self, graph, Hz):
        """
        Build residual r = -div( mu * grad Hz )
        graph.x = [x,y,mu]
        Hz: [N,1]
        returns r: [N,1]
        """
        pos = graph.pos
        mu  = graph.x[:,2:3]    # [N,1]

        # 1) fac ∂Hz/∂pos  → grad_Hz: [N,2]
        grad_Hz = torch.autograd.grad(
            Hz, pos,
            grad_outputs=torch.ones_like(Hz),
            create_graph=True,   # <— no graph!
        )[0]

        # 2) flux = mu * grad_Hz         [N,2]
        flux = mu * grad_Hz

        # 3) divergence: ∑_i ∂flux_i/∂pos_i
        div = 0
        for i in range(2):
            div_i = torch.autograd.grad(
                flux[:,i:i+1], pos,
                grad_outputs=torch.ones_like(flux[:,i:i+1]),
                create_graph=True,   
            )[0][:,i:i+1]
            div = div + div_i

        # PDE: -div = 0
        return -div
        

    
    

    
    
