# PyZX - Python library for quantum circuit rewriting 
#        and optimisation using the ZX-calculus
# Copyright (C) 2018 - Aleks Kissinger and John van de Wetering

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Nothing in this file will make sense if you haven't read Section IV of
# https://journals.aps.org/prx/pdf/10.1103/PhysRevX.6.021043
# In particular the text below equation (10) and equation (11) itself.

import math
sq2 = math.sqrt(2)
from fractions import Fraction

from . import simplify

MAGIC_GLOBAL = -(7+5*sq2)/(2+2j)
MAGIC_B60 = -16 + 12*sq2
MAGIC_B66 = 96 - 68*sq2
MAGIC_E6 = 10 - 7*sq2
MAGIC_O6 = -14 + 10*sq2
MAGIC_K6 = 7 - 5*sq2
MAGIC_PHI = 10 - 7*sq2

class SumGraph(object):
    """Container class for a sum of ZX-diagrams"""
    def __init__(self, graphs=None):
        if graphs:
            self.graphs = graphs
        else:
            self.graphs = []
            
    def to_tensor(self):
        if not self.graphs: return 0
        t = self.graphs[0].to_tensor(True)
        for i in range(len(self.graphs)-1):
            t = t + self.graphs[i+1].to_tensor(True)
        return t

    def to_matrix(self):
        if not self.graphs: return 0
        t = self.graphs[0].to_matrix(True)
        for i in range(len(self.graphs)-1):
            t = t + self.graphs[i+1].to_matrix(True)
        return t

    def full_reduce(self, quiet=True):
        for i, g in enumerate(self.graphs):
            if not quiet:
                print("Graph {:d}:".format(i))
            simplify.full_reduce(g, quiet=quiet)

def replace_magic_states(g):
    """This function takes in a ZX-diagram in graph-like form 
    (all spiders fused, only Z spiders, only H-edges between spiders),
    and splits it into a sum over smaller diagrams by using the magic
    state decomposition of Bravyi, Smith, and Smolin (2016), PRX 6, 021043.
    """
    g = g.copy() # We copy here, so that the vertex labels we get will be the same ones if we copy the graph again
    phases = g.phases()

    # First we find 6 T-like spiders
    states = []
    gadgets = []
    for v in g.vertices():
        if not phases[v] or phases[v].denominator != 4: continue
        if len(g.neighbours(v)) == 1:
            w = list(g.neighbours(v))[0]
            #if g.type(w) == 1:
            gadgets.append(v)
        else:
            states.append(v)

    if len(states) >= 6:
        candidates = states[:6]
    else:
        candidates = states.copy()
        candidates.extend(gadgets[:6-len(states)])

    graphs = []
    replace_functions = [replace_B60, replace_B66, replace_E6, replace_O6, replace_K6, replace_phi1, replace_phi2]
    for func in replace_functions:
        h = func(g.copy(), candidates)
        h.scalar.add_float(MAGIC_GLOBAL)
        graphs.append(h)

    return SumGraph(graphs)

def replace_B60(g, verts):
    g.scalar.add_float(MAGIC_B60)
    g.scalar.add_power(-6)
    for v in verts:
        g.add_to_phase(v,Fraction(-1,4))
    return g

def replace_B66(g, verts):
    g.scalar.add_float(MAGIC_B66)
    g.scalar.add_power(-6)
    g.scalar.add_phase(Fraction(1))
    for v in verts:
        g.add_to_phase(v,Fraction(-1,4))
        g.add_to_phase(v,Fraction(1))
    return g

def replace_E6(g, verts):
    g.scalar.add_float(MAGIC_E6)
    g.scalar.add_power(4)
    g.scalar.add_phase(Fraction(1,2))
    av = 0
    for v in verts:
        g.add_to_phase(v,Fraction(-1,4))
        g.add_to_phase(v, Fraction(1,2))
        av += g.row(v)
    w = g.add_vertex(1,-1,av/6,Fraction(1))
    g.add_edges([(v,w) for v in verts],2)
    return g

def replace_O6(g, verts):
    g.scalar.add_float(MAGIC_O6)
    g.scalar.add_power(4)
    g.scalar.add_phase(Fraction(1,2))
    av = 0
    for v in verts:
        g.add_to_phase(v,Fraction(-1,4))
        g.add_to_phase(v, Fraction(1,2))
        av += g.row(v)
    w = g.add_vertex(1,-1,av/6,Fraction(0))
    g.add_edges([(v,w) for v in verts],2)
    return g

def replace_K6(g, verts):
    g.scalar.add_float(MAGIC_K6)
    g.scalar.add_power(5)
    g.scalar.add_phase(Fraction(1,4))
    av = 0
    for v in verts:
        g.add_to_phase(v,Fraction(-1,4))
        av += g.row(v)
    w = g.add_vertex(1,-1,av/6,Fraction(3,2))
    g.add_edges([(v,w) for v in verts],1)
    return g

def replace_phi1(g, verts):
    g.scalar.add_float(MAGIC_PHI)
    g.scalar.add_power(9)
    g.scalar.add_phase(Fraction(3,2))
    w6 = g.add_vertex(1,-1, g.row(verts[5])+0.5, Fraction(1))
    g.add_to_phase(verts[5],Fraction(-1,4))
    g.add_edge((verts[5],w6))
    ws = []
    for v in verts[:-1]:
        g.add_to_phase(v,Fraction(-1,4))
        w = g.add_vertex(1,-1, g.row(v)+0.5)
        g.add_edges([(w6,w),(v,w)],2)
        ws.append(w)
    w1,w2,w3,w4,w5 = ws
    g.add_edges([(w1,w3),(w1,w4),(w2,w4),(w2,w5),(w3,w5)],2)
    return g

def replace_phi2(g, verts):
    v1,v2,v3,v4,v5,v6 = verts
    verts = [v1,v2,v4,v5,v6,v3]
    return replace_phi1(g,verts)