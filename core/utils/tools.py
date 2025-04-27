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
    model     = config.model
    optimizer = config.optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lrstep, gamma=0.99
    )
    physics   = config.func_main
    tol       = physics.bc_tol
    device    = next(model.parameters()).device

    # 1) load and send graph to device
    graph = config.graph.to(device)
    graph = physics.graph_modify(graph)

    # 2) masks: left‐BC, top/bottom for Neumann, & single interface
    x = graph.pos[:, 0:1]
    y = graph.pos[:, 1:2]
    dx = x - physics.cx
    dy = y - physics.cy
    r  = torch.sqrt(dx*dx + dy*dy)

    left      = torch.isclose(x, torch.zeros_like(x), atol=tol).squeeze()
    bottom    = torch.isclose(y, torch.zeros_like(y), atol=tol).squeeze()
    top       = torch.isclose(y, torch.ones_like(y),  atol=tol).squeeze()
    interface = ((r > physics.r1 - tol) & (r < physics.r1 + tol)).squeeze()

    # precompute normal on interface
    normals = torch.zeros_like(graph.pos)
    normals[interface] = torch.cat(
        [dx[interface]/r[interface], dy[interface]/r[interface]], dim=1
    )

    # sanity counts
    print(f"Sanity: left‐BC={int(left.sum())}, "
          f"bottom‐BC={int(bottom.sum())}, top‐BC={int(top.sum())}, "
          f"interface={int(interface.sum())} nodes")

    # 3) get trainable params
    params = [p for p in model.parameters() if p.requires_grad]
            
    N_tot = graph.pos.shape[0]
    M_if  = interface.sum().float()   # number of interface points
    M_neu  = bottom.sum().float() + top.sum().float()      # number of Neu points
    # 4) training loop
    for epoch in range(1, config.epchoes+1):
        optimizer.zero_grad()

        # a) forward + PDE
        raw    = model(graph)                    # [N,1]
        u_hat  = physics._ansatz_u(graph, raw)
        r_pde, grad_u = physics.pde_residual(graph, u_hat)
        loss_pde = torch.mean(r_pde**2)

        # b) interface‐jump
        eps_in, eps_out = physics.eps_inner, physics.eps_outer
        gi    = grad_u[interface]
        n     = normals[interface]
        jump  = (eps_in - eps_out) * (gi * n).sum(dim=1)
        loss_if = torch.mean(jump**2) if interface.any() else torch.tensor(0.0, device=device)

        # d) top/bottom Neumann: ∂u/∂n = 0 → here simply ∂u/∂y = 0
        #    grad_u[:,1] is ∂u/∂y
        dy_vals     = grad_u[:, 1]
        top_vals    = dy_vals[top]
        bottom_vals = dy_vals[bottom]
        loss_neu_top    = torch.mean(top_vals**2)    if top.any()    else torch.tensor(0.0, device=device)
        loss_neu_bottom = torch.mean(bottom_vals**2) if bottom.any() else torch.tensor(0.0, device=device)
        loss_neu = loss_neu_top + loss_neu_bottom

        
        


        # f) total loss
 
        L = loss_pde + loss_if  + loss_neu
        if epoch % 100 == 0:
            print(f"[{epoch:4d}] PDE={loss_pde:.3e}  "
                  f"IF={loss_if:.3e}  "
                  f"NEU={loss_neu:.3e}")

        # g) backward & step
        L.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

    # save & finish
    model.save_model(optimizer)
    print("Done. Final loss:", L.item())


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
