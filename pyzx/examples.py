from .graph import Graph

def complimentarity():
    g = ig.Graph()
    g.add_vertex(t='B')
    g.add_vertex(t='Z',phase='0.0')
    g.add_vertex(t='X',phase='0.0')
    g.add_vertex(t='B')
    g.add_edges([(0,1),(1,2),(1,2),(2,3)])
    return g

def cnots():
	g = ig.Graph()


def zigzag(sz, backend=None):
    g = Graph(backend)
    g.add_vertices(2*sz+4)
    for i in range(1,sz+1):
        g.set_type(2*i, (i%2)+1)
        g.set_type(2*i+1, (i%2)+1)
    g.add_edges([(0,2),(1,3)])
    g.add_edges([(2*i,2*i+2) for i in range(1,sz)])
    g.add_edges([(2*i,2*i+3) for i in range(1,sz)])
    g.add_edges([(2*i+1,2*i+2) for i in range(1,sz)])
    g.add_edges([(2*i+1,2*i+3) for i in range(1,sz)])
    g.add_edges([(2*sz,2*sz+2),(2*sz+1,2*sz+3)])
    return g

def zigzag2(sz, backend=None):
    g = Graph(backend)
    g.add_vertices(2*sz+4)
    for i in range(1,sz+1):
        g.set_type(2*i, ((i//2)%2)+1)
        g.set_type(2*i+1, ((i//2)%2)+1)
    g.add_edges([(0,2),(1,3)])
    g.add_edges([(2*i,2*i+2) for i in range(1,sz)])
    g.add_edges([(2*i+1,2*i+3) for i in range(1,sz)])
    g.add_edges([(4*i+1+2,4*i+2+2) for i in range(0,sz//2)])
    g.add_edges([(4*i+2,4*i+3+2) for i in range(0,sz//2)])
    g.add_edges([(2*sz,2*sz+2),(2*sz+1,2*sz+3)])
    return g

def t_to_zx(g):
	'''takes a graph where the 't' attributes are ints, and turns it into 'Z', 'X', or 'B' '''
	names = ['B', 'Z', 'X']
	for v in g.vs:
		if not v['t']: v['t'] = 'B'
		else: v['t'] = names[v['t']]
	return g