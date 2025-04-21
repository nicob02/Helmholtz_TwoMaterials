import json
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import math

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
    best_loss  = np.inf
    # 1) Build fixed node features once: [x, y, f]
    x = graph.pos[:,0:1]
    y = graph.pos[:,1:2]
    f = (
    2 * math.pi * torch.cos(math.pi * y) * torch.sin(math.pi * x)
    + 2 * math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y)
    + (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
    - 2 * (math.pi ** 2) * (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
    )
    
    
    for epoch in range(1, config.epchoes + 1):  # Creates different ic and solves the problem, does this epoch # of times
        
        graph.x = torch.cat([graph.pos, f], dim=-1)  # shape [N,3]
        u_raw = model(graph)  

        # 3) Enforce Dirichlet BC = 0 via ansatz (or hard clamp)
        u = config.bc1(graph, u_raw)

        # 4) Compute PDE residual: -Δ u + u - f
        res = config.pde(graph, values_this=u)       # uses laplacian_ad internally
        loss = torch.norm(res)                       # L2 norm of residual
    
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
    """
    Single‐shot evaluation of the trained steady‐state GNN.
    Returns:
      u_pred (numpy array [N,1]): Predicted solution at each mesh node.
    """
    # 1) Move model and graph to the right device
    model = config.model.to(config.device)
    graph = config.graph.to(config.device)

    # 2) Build the fixed node features [x, y, f] exactly as in training
    x = graph.pos[:, 0:1]
    y = graph.pos[:, 1:2]
    f = (
        2 * math.pi * torch.cos(math.pi * y) * torch.sin(math.pi * x)
      + 2 * math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y)
      + (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
      - 2 * (math.pi**2) * (x + y) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
    )
    graph.x = torch.cat([x, y, f], dim=-1)

    # 3) Forward pass + boundary enforcement
    u_raw  = model(graph)               # shape [N,1]
    u_pred = config.bc1(graph, u_raw)   # apply ansatz/hard clamp

    return u_pred.cpu().numpy()


def compute_steady_error(u_pred, u_exact, config):
    """
    Compute the relative L2 error between predicted and exact solutions.
    Returns:
      rel_l2_error (float)
      u_exact      (numpy array [N,1])
    """
    # Relative L2 norm: ||u_pred - u_exact||_2 / ||u_exact||_2
    num = np.linalg.norm(u_pred - u_exact)
    den = np.linalg.norm(u_exact)
    rel_l2 = num/den
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
