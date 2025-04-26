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
    graph     = config.graph
    optimizer = config.optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=config.lrstep,
                                                gamma=0.99)
    physics   = config.func_main
    device    = next(model.parameters()).device
    tol       = physics.bc_tol

    # 1) Build static features once
    graph = physics.graph_modify(graph)

    # 2) Precompute masks / normals
    x = graph.pos[:, 0:1]
    y = graph.pos[:, 1:2]
    left = torch.isclose(x, torch.zeros_like(x), atol=tol).squeeze()
    # (we won’t use right/top/bottom since we’re only doing left‐BC here)
    dx = x - physics.cx;  dy = y - physics.cy
    r  = torch.sqrt(dx*dx + dy*dy)
    if1 = ((r > physics.r1 - tol) & (r < physics.r1 + tol)).squeeze()
    if2 = ((r > physics.r2 - tol) & (r < physics.r2 + tol)).squeeze()
    rad_normals = torch.zeros_like(graph.pos, device=device)
    rad_normals[if1] = torch.cat([dx[if1]/r[if1], dy[if1]/r[if1]], dim=1)
    rad_normals[if2] = torch.cat([dx[if2]/r[if2], dy[if2]/r[if2]], dim=1)

    params = [p for p in model.parameters() if p.requires_grad]

    eps = 1e-8
    for epoch in range(1, config.epchoes + 1):
        # ---- forward ----
        u_hat     = model(graph)                          # raw network output
        r_pde, grad_u = physics.pde_residual(graph, u_hat)

        # PDE loss
        loss_pde = torch.mean(r_pde**2)

        # Interface losses (two rings → combined)
        eps1, eps2, eps3 = physics.eps1, physics.eps2, physics.eps3

        # inner circle
        gi      = grad_u[if1]
        n1      = rad_normals[if1]
        jump1   = (eps1 - eps2)*(gi * n1).sum(dim=1)
        loss_if1 = torch.mean(jump1**2) if if1.any() else torch.tensor(0.0, device=device)

        # outer circle
        gj      = grad_u[if2]
        n2      = rad_normals[if2]
        jump2   = (eps2 - eps3)*(gj * n2).sum(dim=1)
        loss_if2 = torch.mean(jump2**2) if if2.any() else torch.tensor(0.0, device=device)

        loss_if = loss_if1 + loss_if2

        # Left‐boundary Dirichlet: u(0,y)=0
        loss_bc = torch.mean(u_hat[left]**2)

        # ---- compute gradient‐norms ----
        # note: retain_graph so we can do multiple grad calls before final backward
        grads_pde = torch.autograd.grad(loss_pde,
                                        params,
                                        retain_graph=True,
                                        create_graph=True)
        G_pde = torch.sqrt(sum((g**2).sum() for g in grads_pde) + eps)

        grads_if = torch.autograd.grad(loss_if,
                                       params,
                                       retain_graph=True,
                                       create_graph=True)
        G_if  = torch.sqrt(sum((g**2).sum() for g in grads_if) + eps)

        grads_bc = torch.autograd.grad(loss_bc,
                                       params,
                                       retain_graph=True,
                                       create_graph=True)
        G_bc  = torch.sqrt(sum((g**2).sum() for g in grads_bc) + eps)

        # ---- NTK‐inspired weights ----
        lambda_if = (loss_pde.detach() / (loss_if.detach() + eps)) * (G_pde.detach() / (G_if.detach() + eps))
        lambda_bc = (loss_pde.detach() / (loss_bc.detach() + eps)) * (G_pde.detach() / (G_bc.detach() + eps))

        # clamp to avoid extremes
        lambda_if = torch.clamp(lambda_if, min=1e-3, max=1e3)
        lambda_bc = torch.clamp(lambda_bc, min=1e-3, max=1e3)

        # ---- assemble total loss & step ----
        loss = loss_pde + lambda_if * loss_if + lambda_bc * loss_bc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            print(f"[Epoch {epoch:4d}] "
                  f"PDE={loss_pde:.3e}  IF={loss_if:.3e}  BC={loss_bc:.3e}  "
                  f"λ_if={lambda_if:.2f}  λ_bc={lambda_bc:.2f}  "
                  f"Total={loss:.3e}")

    model.save_model(optimizer)
    print("Training completed!  Final loss = %.3e" % loss.item())

@torch.no_grad()
def modelTester(config):
    model   = config.model.to(config.device).eval()
    graph   = config.graph.to(config.device)
    physics = config.func_main
    
    graph = physics.graph_modify(graph)
    u_hat     = model(graph)                           # [N,1]
   # u_hat   = physics._ansatz_u(graph, raw)          # enforce Dirichlet @ x=0
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
