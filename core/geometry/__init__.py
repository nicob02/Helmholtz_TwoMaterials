from fenics import Point
from mshr import generate_mesh, Rectangle, Circle
import numpy as np
from enum import IntEnum
import torch_geometric.transforms as T
from torch_geometric.data import Data
import torch
from dolfin import *

class NodeType(IntEnum):
    inner=0
    boundary=1

def get_node_type(pos, radius_ratio=None):
    min_x, max_x = pos[:,0].min(), pos[:,0].max()
    min_y, max_y = pos[:,1].min(), pos[:,1].max()
    on_left   = np.isclose(pos[:,0], min_x)
    on_right  = np.isclose(pos[:,0], max_x)
    on_bottom = np.isclose(pos[:,1], min_y)
    on_top    = np.isclose(pos[:,1], max_y)
    on_bnd    = on_left|on_right|on_bottom|on_top

    t = np.ones((pos.shape[0],), dtype=np.int64)*NodeType.inner
    t[on_bnd] = NodeType.boundary
    return t
    

class ElectrodeMesh():
    
    node_type_ref = NodeType
    def __init__(self, density=65, lb=(0, 0), ru=(1, 1)) -> None:
        
        self.transform = T.Compose([
            T.FaceToEdge(remove_faces=False), 
            T.Cartesian(norm=False), 
            T.Distance(norm=False)
            ])
    
        domain = Rectangle(Point(lb[0],lb[1]), Point(ru[0], ru[1]))  # Geometry Domain 
        self.mesh = generate_mesh(domain, density)
        self.pos = self.mesh.coordinates().astype(np.float32)
        self.faces = self.mesh.cells().astype(np.int64).T        
        self.node_type = get_node_type(self.pos).astype(np.int64)
        print("Node numbers: %d"%self.pos.shape[0])
        
    def getGraphData(self):
        graph = Data(pos=torch.as_tensor(self.pos), 
                    face=torch.as_tensor(self.faces))
        graph = self.transform(graph)
        graph.num_nodes = graph.pos.shape[0]
        graph.node_type = torch.as_tensor(self.node_type)
        return graph

