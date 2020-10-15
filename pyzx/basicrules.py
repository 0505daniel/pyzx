# PyZX - Python library for quantum circuit rewriting 
#        and optimization using the ZX-calculus
# Copyright (C) 2018 - Aleks Kissinger and John van de Wetering

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# A collection of rules more easily applied interactively to ZX diagrams.

from .graph.base import BaseGraph, VT
from .utils import VertexType, EdgeType

def color_change_diagram(g: BaseGraph):
    for v in g.vertices():
        if g.type(v) == VertexType.BOUNDARY:
            if g.vertex_degree(v) != 0: raise ValueError("Boundary should only have 1 neighbor.")
            v1 = next(iter(g.neighbors(v)))
            e = g.edge(v,v1)
            g.set_edge_type(e, EdgeType.SIMPLE
                    if g.edge_type(e) == EdgeType.HADAMARD
                    else EdgeType.HADAMARD)
        elif g.type(v) == VertexType.Z:
            g.set_type(v, VertexType.X)
        elif g.type(v) == VertexType.X:
            g.set_type(v, VertexType.Z)

def check_copy_X(g: BaseGraph, v: VT) -> bool:
    if not (g.vertex_degree(v) == 1 and
            g.type(v) == VertexType.X and
            (g.phase(v) == 0 or g.phase(v) == 1)):
        return False
    nv = next(iter(g.neighbors(v)))
    if not (g.type(nv) == VertexType.Z and
            g.edge_type(g.edge(v,nv)) == EdgeType.SIMPLE):
        return False
    return True

def copy_X(g: BaseGraph, v: VT) -> bool:
    if not check_copy_X(g, v): return False    
    nv = next(iter(g.neighbors(v)))
    
    for v1 in g.neighbors(nv):
        if v1 != v:
            q = (2*g.qubit(nv) + g.qubit(v1))/3
            r = (2*g.row(nv) + g.row(v1))/3
            v2 = g.add_vertex(VertexType.Z
                    if g.edge_type(g.edge(nv,v1)) == EdgeType.HADAMARD
                    else VertexType.X, qubit=q, row=r)
            g.add_edge((v2,v1))

    g.remove_vertex(v)
    g.remove_vertex(nv)
    
    return True

def check_copy_Z(g: BaseGraph, v: VT) -> bool:
    color_change_diagram(g)
    b = check_copy_X(g, v)
    color_change_diagram(g)
    return b

def copy_Z(g: BaseGraph, v: VT) -> bool:
    color_change_diagram(g)
    b = copy_X(g, v)
    color_change_diagram(g)
    return b

def check_fuse(g: BaseGraph, v1: VT, v2: VT) -> bool:
    if not (g.connected(v1,v2) and
            ((g.type(v1) == VertexType.Z and g.type(v2) == VertexType.Z) or
             (g.type(v1) == VertexType.X and g.type(v2) == VertexType.X)) and
            g.edge_type(g.edge(v1,v2)) == EdgeType.SIMPLE):
        return False
    else:
        return True

def fuse(g: BaseGraph, v1: VT, v2: VT) -> bool:
    if not check_fuse(g, v1, v2): return False
    g.add_to_phase(v1, g.phase(v2))
    for v3 in g.neighbors(v2):
        if v3 != v1:
            g.add_edge_smart((v1,v3), edgetype=g.edge_type(g.edge(v2,v3)))
    
    g.remove_vertex(v2)
    return True

def remove_id(g: BaseGraph, v: VT) -> bool:
    if not (g.vertex_degree(v) == 2 and g.phase(v) == 0):
        return False
    
    v1, v2 = tuple(g.neighbors(v))
    g.add_edge_smart((v1,v2), edgetype=EdgeType.SIMPLE
            if g.edge_type(g.edge(v,v1)) == g.edge_type(g.edge(v,v2))
            else EdgeType.HADAMARD)
    g.remove_vertex(v)
    
    return True



