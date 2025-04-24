import torch
from core.utils.tools import parse_config, modelTester, RemoveDir
from core.utils.tools import compute_steady_error, render_results
from core.models import msgPassing
import os
from functions import MagneticFunc as Func
import matplotlib.pyplot as plt
from core.geometry import ElectrodeMesh
out_ndim = 1

dens=65
ckptpath = 'checkpoint/simulator_%s.pth' % Func.func_name    #FIGURE THIS OUT
device = torch.device(0)

func_main = Func(mu_in=3.0, mu_out=1.0,
                    center=(0.5,0.5), radius=0.2, steep=500.0)

mesh = ElectrodeMesh(ru=(1, 1), lb=(0, 0), density=65)
graph = mesh.getGraphData()
model = msgPassing(message_passing_num=3, node_input_size=out_ndim+2, 
                   edge_input_size=3, ndim=out_ndim, device=device, model_dir=ckptpath)
model.load_model(ckptpath)
model.to(device)
model.eval()
test_steps = 20

test_config = parse_config()

#model = kwargs['model'] # Extracts the model's dictioanry with the weights and biases values
setattr(test_config, 'device', device)   
setattr(test_config, 'model', model)
setattr(test_config, 'test_steps', test_steps)
setattr(test_config, 'NodeTypesRef', ElectrodeMesh.node_type_ref)
setattr(test_config, 'ndim', out_ndim)
setattr(test_config, 'graph_modify', func_main.graph_modify)
setattr(test_config, 'graph', graph)
setattr(test_config, 'density', dens)
setattr(test_config, 'func_main', func_main)
      

#-----------------------------------------

print('************* model test starts! ***********************')
H_z = modelTester(test_config)

pos_np = graph.pos.cpu().numpy()
x, y   = pos_np[:,0], pos_np[:,1]

fig, axes = plt.subplots(1, 2, figsize=(12,5), tight_layout=True)

# Hz
sc0 = axes[0].scatter(x, y, c=H_z.flatten(), cmap='viridis', s=5)
axes[0].set_title("Predicted H_z")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
plt.colorbar(sc0, ax=axes[0], shrink=0.7)

plt.savefig("H_z.png", dpi=300)
plt.close(fig)
print("Done — predictions plotted to H_z.png")
