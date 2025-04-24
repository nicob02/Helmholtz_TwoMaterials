import json
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import math
from torch_geometric.data import Data

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
    model  = config.model
    graph  = config.graph
    physics= config.func_main
    opt    = config.optimizer
    sched  = torch.optim.lr_scheduler.StepLR(opt,
                    step_size=config.lrstep, gamma=0.99)

    # 1) build static node features (mu, etc.)
    graph = physics.graph_modify(graph)

    for epoch in range(1, config.epochs+1):
        raw = model(graph)                         # [N,1] raw Hz
        Hz  = physics._ansatz_Hz(graph, raw)       # hard‐BC
        
        # 2) PDE residual
        r   = physics.pde_residual(graph, Hz)      # [N,1]
        loss= torch.mean(r**2)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if epoch % config.log_every == 0:
            print(f"[Epoch {epoch:5d}]  loss = {loss:.3e}")

    # save
    model.save_model(opt)
    print("Training done.")

@torch.no_grad()
def modelTester(config):
    model   = config.model.to(config.device).eval()
    graph   = config.graph.to(config.device)
    physics = config.func_main
    
    graph = physics.graph_modify(graph)
    raw   = model(graph)
    Hz    = physics._ansatz_Hz(graph, raw)
    return Hz.cpu().numpy()      # shape [N,1]

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

def render_results(u_pred, u_exact, graph, filename="steady_results.png"):
    """
    Scatter‐plot Exact, Predicted, and Absolute Error on the mesh nodes.
    """
    pos = graph.pos.cpu().numpy()
    x, y = pos[:,0], pos[:,1]
    error = np.abs(u_exact - u_pred)

    fig, axes = plt.subplots(1, 3, figsize=(18,5))

    # 1) Exact
    sc0 = axes[0].scatter(x, y, c=u_exact.flatten(), cmap='viridis', s=5)
    axes[0].set_title("Exact Solution")
    plt.colorbar(sc0, ax=axes[0], shrink=0.7)

    # 2) Predicted
    sc1 = axes[1].scatter(x, y, c=u_pred.flatten(), cmap='viridis', s=5)
    axes[1].set_title("GNN Prediction")
    plt.colorbar(sc1, ax=axes[1], shrink=0.7)

    # 3) Absolute Error
    sc2 = axes[2].scatter(x, y, c=error.flatten(), cmap='magma', s=5)
    axes[2].set_title("Absolute Error")
    plt.colorbar(sc2, ax=axes[2], shrink=0.7)

    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
