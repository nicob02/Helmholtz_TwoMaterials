import json
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import math
from torch_geometric.data import Data
import torch.autograd as autograd

def RemoveDir(filepath):
    '''
    If the folder doesn't exist, create it; and if it exists, clear it.
    '''
    if not os.path.exists(filepath):
        os.makedirs(filepath,exist_ok=True)
    else:
        shutil.rmtree(filepath)
        os.makedirs(filepath, exist_ok=True)


class Config:
    def __init__(self) -> None:
        pass
    def __setattr__(self, __name: str, __value) -> None:
        self.__dict__[__name] = __value


def parse_config(file='config.json'):
    configs = Config() 
    if not os.path.exists(file):
        return configs
    with open(file, 'r') as f:
        data = json.load(f)
        for k, v in data.items():
            config = Config()
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    config.setattr(k1, v1)
            else:
                raise TypeError
            configs[k] = config
    return configs[k]

def modelTrainer(config):
    
    model = config.model
    graph = config.graph
    scheduler = torch.optim.lr_scheduler.StepLR(
        config.optimizer, step_size=config.lrstep, gamma=0.99)  

    physics   = config.func_main
    tol       = physics.bc_tol

    # build static features once
    graph = physics.graph_modify(graph)
    
    # precompute masks and normals for BC & interfaces
    x = graph.pos[:,0:1]
    y = graph.pos[:,1:2]
    # boundary masks
    left   = torch.isclose(x, torch.zeros_like(x), atol=tol).squeeze()
    right  = torch.isclose(x, torch.ones_like(x),  atol=tol).squeeze()
    bottom = torch.isclose(y, torch.zeros_like(y), atol=tol).squeeze()
    top    = torch.isclose(y, torch.ones_like(y),  atol=tol).squeeze()

    # interface masks and radial normals
    dx = x - physics.cx;  dy = y - physics.cy
    r  = torch.sqrt(dx*dx + dy*dy)
    if1 = ((r > physics.r1 - tol) & (r < physics.r1 + tol)).squeeze()
    if2 = ((r > physics.r2 - tol) & (r < physics.r2 + tol)).squeeze()
    rad_normals = torch.zeros_like(graph.pos)
    rad_normals[if1] = torch.cat([dx[if1]/r[if1], dy[if1]/r[if1]], dim=1)
    rad_normals[if2] = torch.cat([dx[if2]/r[if2], dy[if2]/r[if2]], dim=1)
    
    for epoch in range(1, config.epchoes + 1):  # Creates different ic and solves the problem, does this epoch # of times
        
        raw     = model(graph)                           # [N,1]
        u_hat   = physics._ansatz_u(graph, raw)          # enforce Dirichlet @ x=0
        r_pde, grad_u = physics.pde_residual(graph, u_hat)

        # PDE loss
        loss_pde = torch.mean(r_pde**2)

        # interface‐flux continuity
        # flux‐jump at r1: (eps1−eps2)*(n·∇u)
        eps1,eps2,eps3 = physics.eps1,physics.eps2,physics.eps3
        # mask1
        n1 = rad_normals[if1]
        gi = grad_u[if1]
        jump1 = (eps1-eps2)*(gi * n1).sum(dim=1)
        loss_if1 = torch.mean(jump1**2) if if1.any() else 0.0
        # mask2
        n2 = rad_normals[if2]
        gj = grad_u[if2]
        jump2 = (eps2-eps3)*(gj * n2).sum(dim=1)
        loss_if2 = torch.mean(jump2**2) if if2.any() else 0.0

        print("if_loss")
        print(loss_if1 + loss_if2)
        print("pde_loss")
        print(loss_pde)
        loss = ( loss_pde + config.lambda_if  * (loss_if1 + loss_if2))
        
        config.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        config.optimizer.step()
        scheduler.step()
        
        if epoch % 500 == 0:
            print(f"[Epoch {epoch:4d}] Loss = {loss.item():.3e}")
            
    model.save_model(config.optimizer)
    print('model saved at loss: %.4e' % loss)    
    print("Training completed!")

@torch.no_grad()
def modelTester(config):
    model   = config.model.to(config.device).eval()
    graph   = config.graph.to(config.device)
    physics = config.func_main
    
    graph = physics.graph_modify(graph)
    raw     = model(graph)                           # [N,1]
    u_hat   = physics._ansatz_u(graph, raw)          # enforce Dirichlet @ x=0
    return u_hat.cpu().numpy()      # shape [N,1]
def compute_steady_error(u_pred, u_exact, config):
    # 1) Convert predictions to NumPy
    if isinstance(u_pred, torch.Tensor):
        u_pred_np = u_pred.detach().cpu().numpy()
    else:
        u_pred_np = np.array(u_pred, copy=False)

    # 2) Convert exact to NumPy
    if isinstance(u_exact, torch.Tensor):
        u_exact_np = u_exact.detach().cpu().numpy()
    else:
        u_exact_np = np.array(u_exact, copy=False)

    # 3) Flatten both to 1D arrays
    u_pred_flat  = u_pred_np.reshape(-1)
    u_exact_flat = u_exact_np.reshape(-1)

    # 4) Compute relative L2 norm
    num   = np.linalg.norm(u_pred_flat - u_exact_flat)
    denom = np.linalg.norm(u_exact_flat)
    rel_l2 = num / (denom + 1e-16)  # small eps to avoid div0

    return rel_l2

def render_results(u_pred, u_exact, graph, filename="NNvsFEM.png"):
    """
    Scatter-plot Exact, Predicted, and Absolute Error on the mesh nodes.
    """
    # pull out XY
    pos = graph.pos.cpu().numpy()
    x, y = pos[:,0], pos[:,1]

    # ensure both are flat 1-D arrays of length N
    u_pred_flat  = np.array(u_pred).reshape(-1)
    u_exact_flat = np.array(u_exact).reshape(-1)
    assert u_pred_flat.shape == u_exact_flat.shape, "pred/exact length mismatch"

    # now compute error
    error = np.abs(u_exact_flat - u_pred_flat)

    # set up panels
    fig, axes = plt.subplots(1, 3, figsize=(18,5))

    # 1) Exact
    sc0 = axes[0].scatter(x, y, c=u_exact_flat, cmap='viridis', s=5)
    axes[0].set_title("Exact Solution")
    plt.colorbar(sc0, ax=axes[0], shrink=0.7)

    # 2) Predicted
    sc1 = axes[1].scatter(x, y, c=u_pred_flat, cmap='viridis', s=5)
    axes[1].set_title("GNN Prediction")
    plt.colorbar(sc1, ax=axes[1], shrink=0.7)

    # 3) Absolute Error
    sc2 = axes[2].scatter(x, y, c=error, cmap='magma', s=5)
    axes[2].set_title("Absolute Error")
    plt.colorbar(sc2, ax=axes[2], shrink=0.7)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
