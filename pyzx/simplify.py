"""Contains simplification procedures based in the rewrite rules in rules_.
"""

from __future__ import print_function

try:
    import multiprocessing as mp
except ImportError:
    pass

__all__ = ['bialg_simp','spider_simp', 'phase_free_simp', 'pivot_simp', 
        'lcomp_simp', 'clifford_simp', 't_count', 'to_gh', 'to_rg']

from .rules import *

def simp(g, name, match, rewrite):
    """Helper method for generating simplification strategies based on rules in rules_.
    It keeps matching and rewriting with the given methods until it can no longer do so.
    Example usage: ``simp(g, 'spider_simp', rules.match_spider_parallel, rules.spider)``

    :param g: The graph that needs to be simplified.
    :param str name: The name of this rewrite rule.
    :param match: One of the ``match_*`` functions of rules_.
    :param rewrite: One of the rewrite functions of rules_."""
    i = 0
    new_matches = True
    print(name)
    while new_matches:
        i += 1
        new_matches = False
        m = match(g)
        if len(m) > 0:
            print(len(m), end='')
            #print(len(m), end='', flush=True) #flush only supported on Python >3.3
            etab, rem_verts, check_isolated_vertices = rewrite(g, m)
            g.add_edge_table(etab)
            g.remove_vertices(rem_verts)
            if check_isolated_vertices: g.remove_isolated_vertices()
            print('. ', end='')
            #print('. ', end='', flush=True)
            new_matches = True
    print('\nfinished in ' + str(i) + ' iterations')

def pivot_simp(g):
    return simp(g, 'pivot_simp', match_pivot_parallel, pivot)

def lcomp_simp(g):
    return simp(g, 'lcomp_simp', match_lcomp_parallel, lcomp)

def bialg_simp(g):
    return simp(g, 'bialg_simp', match_bialg_parallel, bialg)

def spider_simp(g):
    return simp(g, 'spider_simp', match_spider_parallel, spider)

def id_simp(g):
    return simp(g, 'id_simp', match_ids_parallel, remove_ids)

def phase_free_simp(g):
    '''Performs the following set of simplifications on the graph:
    spider -> bialg'''
    spider_simp(g)
    bialg_simp(g)

def clifford_simp(g):
    '''Performs the following set of simplifications on the graph:
    spider -> pivot -> lcomp -> pivot -> id'''
    spider_simp(g)
    to_gh(g)
    pivot_simp(g)
    lcomp_simp(g)
    pivot_simp(g)
    #to_rg(g)
    id_simp(g)

def to_gh(g):
    ty = g.types()
    for v in g.vertices():
        if ty[v] == 2:
            g.set_type(v, 1)
            for e in g.incident_edges(v):
                et = g.edge_type(e)
                if et == 2: g.set_edge_type(e,1)
                elif et == 1: g.set_edge_type(e,2)

def to_rg(g, select=None):
    """Turn into RG form by colour-changing vertices which satisfy the given predicate.
    By default, the predicate is set to greedily reducing the number of h-edges."""
    if not select:
        select = lambda v: (
            len([e for e in g.incident_edges(v) if g.edge_type(e) == 1]) <
            len([e for e in g.incident_edges(v) if g.edge_type(e) == 2])
            )

    ty = g.types()
    for v in g.vertices():
        if select(v):
            if ty[v] == 1:
                g.set_type(v, 2)
                for e in g.incident_edges(v):
                    g.set_edge_type(e, 1 if g.edge_type(e) == 2 else 2)
            elif ty[v] == 2:
                g.set_type(v, 1)
                for e in g.incident_edges(v):
                    g.set_edge_type(e, 1 if g.edge_type(e) == 2 else 2)




def simp_iter(g, name, match, rewrite):
    i = 0
    new_matches = True
    while new_matches:
        i += 1
        new_matches = False
        m = match(g)
        if len(m) > 0:
            etab, rem_verts, check_isolated_vertices = rewrite(g, m)
            g.add_edge_table(etab)
            g.remove_vertices(rem_verts)
            if check_isolated_vertices: g.remove_isolated_vertices()
            yield g, name+str(i)
            new_matches = True

def pivot_iter(g):
    return simp_iter(g, 'pivot', match_pivot_parallel, pivot)

def lcomp_iter(g):
    return simp_iter(g, 'lcomp', match_lcomp_parallel, lcomp)

def bialg_iter(g):
    return simp_iter(g, 'bialg', match_bialg_parallel, bialg)

def spider_iter(g):
    return simp_iter(g, 'spider', match_spider_parallel, spider)

def id_iter(g):
    return simp_iter(g, 'id', match_ids_parallel, remove_ids)

def clifford_iter(g):
    for d  in spider_iter(g): yield d
    to_gh(g)
    yield g, "to_gh"
    for d in lcomp_iter(g): yield d
    for d in pivot_iter(g): yield d
    to_rg(g)
    yield g, "to_rg"
    for d in id_iter(g): yield d



def t_count(g):
    count = 0
    for a in g.phases().values():
        if a.denominator == 4:
            count += 1
    return count

def _worker(arg):
    match, rewrite, g, kwargname, kwarg = arg
    m =  match(g, **{kwargname:kwarg})
    if m: return (len(m),rewrite(g, m))
    return None

def simp_threaded(g, name, match, rewrite, uses_verts=False,safe=False,skip_unthreaded_pass=False):
    nthreads = 5
    i = 0
    new_matches = True
    sep = int(g.vindex / nthreads) + 1 #TODO: currently only works on graph_s
    pool = mp.Pool(processes=nthreads)
    #Threaded pass
    print("Starting {} with {} threads".format(name,str(nthreads)))
    while new_matches:
        new_matches = False
        if uses_verts:
            chunks = [set(g.vertices_in_range(j*sep,(j+1)*sep)) for j in range(nthreads)]
            for j in range(nthreads-1):
                if chunks[j] & chunks[j+1]:
                    raise Exception("overlapping chunks")
            results = pool.map(_worker, ((match, rewrite, g, "vertexlist", chunks[j]) for j in range(nthreads)),
                                        #set(g.vertices_in_range(j*sep,(j+1)*sep))) for j in range(nthreads)),
                                chunksize=1)
        else:
            chunks = [set(g.edges_in_range(j*sep,(j+1)*sep,safe)) for j in range(nthreads)]
            for j in range(nthreads-1):
                if chunks[j] & chunks[j+1]:
                    raise Exception("overlapping chunks")
            results = pool.map(_worker, ((match, rewrite, g, "edgelist", chunks[j]) for j in range(nthreads)),
                                        #set(g.edges_in_range(j*sep,(j+1)*sep,safe))) for j in range(nthreads)),
                                chunksize=1)

        check_isolated_vertices = False
        for j,r in enumerate(results):
            if not r: continue
            new_matches = True
            amount, (etab, rem_verts, check) = r
            print(amount, end=',')
            if uses_verts and not chunks[j].issuperset(set(rem_verts)):
                raise Exception("Deleting vertices outside of chunk: ", chunks[j], set(rem_verts))
            else:
                for v in rem_verts:
                    if not (j*sep<v<(j+1)*sep):
                        raise Exception("Deleting vertices outside of chunk")
            g.add_edge_table(etab)
            g.remove_vertices(rem_verts)
            check_isolated_vertices = check
        if check_isolated_vertices: g.remove_isolated_vertices() 
        if new_matches: i += 1
        print('. ', end='')
    pool.close()
    #Unthreaded pass
    if not skip_unthreaded_pass:
        new_matches = True
        if i!=0: print("\nUnthreaded pass: ", end='')
        else: print("Unthreaded pass: ", end='') 
        while new_matches:
            new_matches = False
            m = match(g)
            if len(m) > 0:
                print(len(m), end='')
                etab, rem_verts, check_isolated_vertices = rewrite(g, m)
                g.add_edge_table(etab)
                g.remove_vertices(rem_verts)
                if check_isolated_vertices: g.remove_isolated_vertices()
                print('. ', end='')
                new_matches = True
            if new_matches: i += 1
    if i!=0:  print('\nDid ' + str(i) + ' nonzero iterations')
    else: print('Did ' + str(i) + ' nonzero iterations')


def pivot_threaded(g):
    return simp_threaded(g, 'pivot_simp', match_pivot_parallel, pivot, uses_verts=False,safe=True)

def lcomp_threaded_old(g):
    simp_threaded(g, 'lcomp_simp', match_lcomp_parallel, lcomp, uses_verts=True)

def lcomp_threaded(g):
    simp_threaded(g, 'lcomp_simp', match_lcomp_parallel, lcomp, uses_verts=True,skip_unthreaded_pass=True)
    print("Full pass: ",end='')
    nthreads = 5
    pool = mp.Pool(processes=nthreads)
    i = 0
    new_matches = True
    while new_matches:
        i += 1
        new_matches = False
        matches = match_lcomp_parallel(g)
        if len(matches) > 0:
            print(len(matches), end='')
            vs, ns = [], []
            for v, neighbours in matches:
                a = g.phase(v)
                for v2 in neighbours: g.add_to_phase(v2, -a)
                vs.append(v)
                ns.append(neighbours)
            for etab in pool.map(_lcomp_do, (ns[i:i + 10] for i in range(0, len(ns), 10))):
                g.add_edge_table(etab)
            g.remove_vertices(vs)
            print('. ', end='')
            new_matches = True
    pool.close()
    print('\nfinished in ' + str(i) + ' iterations')

def _lcomp_do(matches):
    etab = dict()
    for neighbours in matches:
        for i in range(len(neighbours)):
            for j in range(i+1, len(neighbours)):
                e = (neighbours[i],neighbours[j])
                if (e[0] > e[1]): e = (e[1],e[0])
                if e not in etab: etab[e] = [0,1]
                else: etab[e][1] += 1

    return etab

def bialg_threaded(g):
    return simp_threaded(g, 'bialg_simp', match_bialg_parallel, bialg, uses_verts=False)

def spider_threaded(g):
    return simp_threaded(g, 'spider_simp', match_spider_parallel, spider, uses_verts=False)

def id_threaded(g):
    return simp_threaded(g, 'id_simp', match_ids_parallel, remove_ids, uses_verts=True)

def clifford_threaded(g):
    spider_threaded(g)
    to_gh(g)
    lcomp_threaded(g)
    pivot_threaded(g)
    to_rg(g)
    id_threaded(g)
