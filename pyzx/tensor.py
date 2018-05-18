import numpy as np
np.set_printoptions(suppress=True)
from math import pi

def contract_all(tensors,conns):
    '''
    Contract the tensors inside the list tensors
    according to the connectivities in conns

    Example input:
    tensors = [np.random.rand(2,3),np.random.rand(3,4,5),np.random.rand(3,4)]
    conns = [((0,1),(2,0)), ((1,1),(2,1))]
    returned shape in this case is (2,3,5)
    Taken from https://stackoverflow.com/questions/42034480/efficient-tensor-contraction-in-python
    '''

    ndims = [t.ndim for t in tensors]
    totdims = sum(ndims)
    dims0 = np.arange(totdims)
    # keep track of sublistout throughout
    sublistout = set(dims0.tolist())
    # cut up the index array according to tensors
    # (throw away empty list at the end)
    inds = np.split(dims0,np.cumsum(ndims))[:-1]
    # we also need to convert to a list, otherwise einsum chokes
    inds = [ind.tolist() for ind in inds]

    # if there were no contractions, we'd call
    # np.einsum(*zip(tensors,inds),sublistout)

    # instead we need to loop over the connectivity graph
    # and manipulate the indices
    for (m,i),(n,j) in conns:
        # tensors[m][i] contracted with tensors[n][j]

        # remove the old indices from sublistout which is a set
        sublistout -= {inds[m][i],inds[n][j]}

        # contract the indices
        inds[n][j] = inds[m][i]

    # zip and flatten the tensors and indices
    args = [subarg for arg in zip(tensors,inds) for subarg in arg]

    # assuming there are no multiple contractions, we're done here
    return np.einsum(*args,sublistout)


def Z_to_tensor(arity, phase):
    m = np.zeros([2]*arity, dtype = complex)
    m[(0,)*arity] = 1
    m[(1,)*arity] = np.exp(1j*phase)
    return m

def X_to_tensor(arity, phase):
    m = np.ones(2**arity,dtype = complex)
    for i in range(2**arity):
        if bin(i).count("1")%2 == 0: 
            m[i] += np.exp(1j*phase)
        else:
            m[i] -= np.exp(1j*phase)
    return np.power(np.sqrt(0.5),arity)*m.reshape([2]*arity)

S = Z_to_tensor(2,0.5*np.pi)
Xphase = X_to_tensor(2,0.5*np.pi)

had = np.sqrt(2)*np.exp(-1j*0.25*np.pi) * (S @ Xphase @ S)
#print(had)

import sys
from io import StringIO

def phase_to_number(s):
    old = sys.stdout
    stdout = StringIO()
    sys.stdout = stdout
    s = s.replace("\\pi", "pi")
    exec("print({})".format(s))
    sys.stdout = old
    return float(stdout.getvalue().strip())


def zx_graph_to_tensor(g):
    '''Takes in a igraph.Graph.
    All nodes should have a 't' attribute valued in 'Z', 'X' or 'B' (for boundary)
    Outputs a multidimensional numpy array
    representing the linear map the ZX-diagram implements'''
    tensors = []
    ids = {}

    for v in g.vs:
        if v['t'] == 'Z':
            phase = phase_to_number(v.attributes().get('phase',"0.0"))
            ids[v.index] = len(tensors)
            tensors.append(Z_to_tensor(v.degree(),phase))
        elif v['t'] == 'X':
            phase = phase_to_number(v.attributes().get('phase',"0.0"))
            ids[v.index] = len(tensors)
            tensors.append(X_to_tensor(v.degree(),phase))
        elif v['t'] == 'B':
            pass
        else:
            raise Exception("Wrong type for node '{}'".format(v['t']))

    conns = []
    contraction_count = {i:0 for i in ids}
    for e in g.es:
        if not (e.source in ids and e.target in ids): continue
        conns.append(((ids[e.source],contraction_count[e.source]),
                     (ids[e.target],contraction_count[e.target])))
        contraction_count[e.source] += 1
        contraction_count[e.target] += 1

    return contract_all(tensors, conns)

if __name__ == '__main__':
    import igraph as ig
    g = ig.Graph()
    g.add_vertex(t='B')
    g.add_vertex(t='Z',phase='0.0')
    g.add_vertex(t='X',phase='0.0')
    g.add_vertex(t='B')
    g.add_edges([(0,1),(1,2),(1,2),(2,3)])

    print(zx_graph_to_tensor(g))
