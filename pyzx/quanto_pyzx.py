from .graph.graph import Graph

import json
from fractions import Fraction



print("Loaded module")

def _quanto_value_to_phase(s):
    if not s: return Fraction(0)
    if r'\pi' in s:
        try:
            r = s.replace(r'\pi','').strip()
            if r.startswith('-'): r = "-1"+r[1:]
            if r.startswith('/'): r = "1"+r
            return Fraction(str(r)) if r else Fraction(1)
        except ValueError:
            raise ValueError("Invalid phase '{}'".format(s))
    return s

def _phase_to_quanto_value(p):
    if not p: return ""
    if isinstance(p, Fraction):
        if p.numerator == -1: v = "-"
        elif p.numerator == 1: v = ""
        else: v = str(p.numerator)
        d = "/"+str(p.denominator) if p.denominator!=1 else ""
        return r"{}\pi{}".format(v,d)
    else: return p
    # if not n: return ""
    # s = str(float(n))
    # return s + r"\pi"

def json_to_graph(js):
    j = json.loads(str(js))
    g = Graph('simple')
    v = 0
    names = {}
    for name,attr in j.get('node_vertices',{}).items():
        g.add_vertex()
        g.set_vdata(v,'name',name)
        names[name] = v
        if 'data' in attr:
            d = attr['data']
            if not 'type' in d or d['type'] == 'Z': g.set_type(v,1)
            elif d['type'] == 'X': g.set_type(v,2)
            else: raise TypeError("unsupported type '{}'".format(d['type']))
            if 'value' in d:
                g.set_angle(v,_quanto_value_to_phase(d['value']))
            else:
                g.set_angle(v,0.0)
        else:
            g.set_type(v,1)
            g.set_angle(v,0.0)
        c = attr['annotation']['coord']
        g.set_vdata(v, 'x', c[0])
        g.set_vdata(v, 'y', c[1])
        v += 1
    for name,attr in j.get('wire_vertices',{}).items():
        g.add_vertex()
        g.set_vdata(v,'name',name)
        names[name] = v
        g.set_type(v,0)
        c = attr['annotation']['coord']
        g.set_vdata(v, 'x', c[0])
        g.set_vdata(v, 'y', c[1])
        v += 1
    edges = []
    for edge in j.get('undir_edges',{}).values():
        edges.append((names[edge['src']],names[edge['tgt']]))
    g.add_edges(edges)

    return g

def json_from_graph(g):
    node_vs = {}
    wire_vs = {}
    edges = {}
    names = {}
    freenamesv = ["v"+str(i) for i in range(g.num_vertices())]
    freenamesb = ["b"+str(i) for i in range(g.num_vertices())]
    for v in g.vertices():
        t = g.get_type(v)
        coord = [g.get_vdata(v,'x'),g.get_vdata(v,'y')]
        name = g.get_vdata(v, 'name')
        if not name:
            if t == 0: name = freenamesb.pop(0)
            else: name = freenamesv.pop(0)
        else: 
            try:
                freenamesb.remove(name) if t==0 else freenamesv.remove(name)
            except:
                print("couldn't remove name '{}'".format(name))
        
        names[v] = name
        if t == 0:
            wire_vs[name] = {"annotation":{"boundary":True,"coord":coord}}
        else:
            node_vs[name] = {"annotation": {"coord":coord},"data":{}}
            if t==2: node_vs[name]["data"]["type"] = "X"
            elif t==1:node_vs[name]["data"]["type"] = "Z"
            elif t!=1: raise Exception("Unkown type "+ str(t))
            phase = _phase_to_quanto_value(g.get_angle(v))
            if phase: node_vs[name]["data"]["value"] = phase
            if not node_vs[name]["data"]: del node_vs[name]["data"]
    for i, e in enumerate(g.edges()):
        s,t = g.edge_st(e)
        edges["e"+ str(i)] = {"src": names[s],"tgt": names[t]}

    return json.dumps({"wire_vertices": wire_vs, 
            "node_vertices": node_vs, 
            "undir_edges": edges})

