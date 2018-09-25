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


"""This module implements the Third Order Duplicate and Destroy algorithm
from Luke E Heyfron and Earl T Campbell 2019 Quantum Sci. Technol. 4 015004
available at http://iopscience.iop.org/article/10.1088/2058-9565/aad604/meta"""

from fractions import Fraction

from .circuit import T, S, Z, ZPhase, CZ, CNOT, ParityPhase
from .linalg import Mat2
from .phasepoly import parity_network




class ParityPolynomial:
    def __init__(self,qubits, poly=None):
        self.qubits = qubits
        if poly:
            self.terms = poly.terms.copy()
        else: self.terms = {}
    
    def copy(self):
        return type(self)(self.qubits, self)
    
    def __str__(self):
        l = []
        for t in sorted(self.terms.keys()):
            val = self.terms[t]
            l.append("{!s}{}".format(val if val!=1 else "", "@".join("x{:d}".format(v) for v in sorted(list(t)))))
        return " + ".join(l)
    
    def __repr__(self):
        return str(self)
    
    def add_term(self, term, value):
        term = tuple(sorted(term))
        if term in self.terms:
            self.terms[term] = (self.terms[term] + value) % 8
        else: self.terms[term] = value % 8
        if not self.terms[term]:
            del self.terms[term]
    
    def add_polynomial(self, poly):
        for term, val in poly.terms.items():
            self.add_term(term, val)
    
    def __add__(self, other):
        p = self.copy()
        p.add_polynomial(other)
        return p
    
    def to_par_matrix(self):
        cols = []
        for par, val in self.terms.items():
            col = [1 if i in par else 0 for i in range(self.qubits)]
            for i in range(val): cols.append(col)
        return Mat2(cols).transpose()

class ParitySingle:
    def __init__(self,startval):
        self.par = {startval}
    
    def __str__(self):
        return "@".join("x{:d}".format(i) for i in sorted(self.par))
    
    def __repr__(self):
        return str(self)
    
    def add_par(self, other):
        self.par.symmetric_difference_update(other.par)


class MultiLinearPoly:
    def __init__(self):
        self.l = {}
        self.q = {}
        self.c = set()
    
    def add_parity(self, par, subtract=False):
        p = []
        mult = -1 if subtract else 1
        for i,v in enumerate(par):
            if v: p.append(i)
        for a in range(len(p)):
            v1 = p[a]
            if v1 not in self.l: self.l[v1] = mult
            else: self.l[v1] = (self.l[v1] + mult) % 8
            
            for b in range(a+1, len(p)):
                v2 = p[b]
                if (v1,v2) not in self.q: self.q[(v1,v2)] = 1 if subtract else 3
                else: self.q[(v1,v2)] = (self.q[(v1,v2)] - mult) % 4
                    
                for c in range(b+1, len(p)):
                    v3 = p[c]
                    if (v1,v2,v3) not in self.c: self.c.add((v1,v2,v3))
                    else: self.c.remove((v1,v2,v3))
    
    def add_par_matrix(self, a, subtract=False):
        for col in a.transpose().data:
            self.add_parity(col,subtract=subtract)
    
    def to_clifford(self):
        gates = []
        for t, v in self.l.items():
            if v == 2:
                gates.append(S(t,adjoint=False))
            elif v == 4:
                gates.append(Z(t))
            elif v == 6:
                gates.append(S(t,adjoint=True))
            elif v != 0:
                raise ValueError("PhasePoly is not Clifford")
        for (t1,t2), v in self.q.items():
            if v == 2:
                gates.append(CZ(t1,t2))
            elif v != 0:
                raise ValueError("PhasePoly is not Clifford")
        if self.c:
            raise ValueError("PhasePoly is not Clifford")
        return gates


def par_matrix_to_gates(a):
    gates = []
    phase = Fraction(1,4)
    for col in a.transpose().data:
        targets = [i for i,v in enumerate(col) if v]
        if len(targets) == 1:
            gates.append(T(targets[0]))
        else:
            gates.append(ParityPhase(phase, *targets))
    return gates

def phase_gates_to_poly(gates, qubits):
    phase_poly = ParityPolynomial(qubits)
    expression_polys = []
    for i in range(qubits):
        expression_polys.append(ParitySingle(i))
    
    for g in gates:
        if isinstance(g, ZPhase):
            par = expression_polys[g.target].par
            phase_poly.add_term(par, int(g.phase*4))
        elif isinstance(g, CZ):
            tgt, ctrl = g.target, g.control
            par1 = expression_polys[tgt].par
            par2 = expression_polys[ctrl].par
            phase_poly.add_term(par1, 2)
            phase_poly.add_term(par2, 2)
            phase_poly.add_term(par1.symmetric_difference(par2), 6)
        elif isinstance(g, CNOT):
            tgt, ctrl = g.target, g.control
            expression_polys[tgt].add_par(expression_polys[ctrl])
        else:
            raise TypeError("Unknown gate type {}".format(str(g)))
    
    return phase_poly, expression_polys




def xi(m, z):
    rows = m.rows()
    data = []
    for alpha in range(rows):
        ra = m.data[alpha]
        for beta in range(alpha+1, rows):
            rb = m.data[beta]
            rab = [i*j for i,j in zip(ra,rb)]
            for gamma in range(beta+1, rows):
                rg = m.data[gamma]
                rag = [i*j for i,j in zip(ra,rg)]
                rbg = [i*j for i,j in zip(rb,rg)]
                
                if z[alpha]:
                    if not z[beta]:
                        if not z[gamma]:
                            data.append(rbg)
                            continue
                        data.append([0 if v1==v2 else 1 for v1,v2 in zip(rbg,rab)])
                        continue
                    elif not z[gamma]:
                        data.append([0 if v1==v2 else 1 for v1,v2 in zip(rbg,rag)])
                        continue
                    else: #z[alpha], z[beta] and z[gamma] are all true
                        r = [0 if v1==v2 else 1 for v1,v2 in zip(rab,rag)]
                        data.append([0 if v1==v2 else 1 for v1,v2 in zip(r,rbg)])
                        continue
                elif z[beta]:
                    if z[gamma]:
                        data.append([0 if v1==v2 else 1 for v1,v2 in zip(rab,rag)])
                        continue
                    data.append(rag)
                    continue
                elif z[gamma]:
                    data.append(rab.copy())
                    continue
    for r in m.data: data.append(r.copy())            
    return Mat2(data)

def find_todd_match(m):
    rows = m.rows()
    cols = m.cols()
    for a in range(cols):
        for b in range(a+1, cols):
            z = [0]*rows
            for i in range(rows):
                r = m.data[i]
                if r[a]:
                    if not r[b]:
                        z[i] = 1
                else:
                    if r[b]:
                        z[i] = 1
            bigm = xi(m, z)
            #print(bigm, '.')
            options = bigm.nullspace(should_copy=False)
            #print(bigm)
            for y in options:
                if y[a] + y[b] == 1: return a,b,z,y

    return -1,-1,None,None


def remove_trivial_cols(m):
    while True:
        newcols = m.rows()
        for a in range(newcols):
            if not any(m.data[a]):
                m.data.pop(a)
                break
            should_break = False
            for b in range(a+1, newcols):
                if m.data[a] == m.data[b]:
                    m.data.pop(b)
                    m.data.pop(a)
                    should_break = True
                    break
            if should_break: break
        else: # Didn't break out of for-loop so didn't find any match
            break
    return newcols

def do_todd_single(m):
    startcols = m.cols()
    a,b,z,y = find_todd_match(m)
    if not z: return m, 0
    m = m.transpose()
    #odd_y = sum(y) % 2
    for i,c in enumerate(m.data):
        if not y[i]: continue
        for j in range(len(c)):
            if z[j]: c[j] = 0 if c[j] else 1
    if sum(y) % 2 == 1:
        m.data.append(z)
    m.data.pop(b)
    m.data.pop(a)
    
    newcols = remove_trivial_cols(m)
                
    return m.transpose(), startcols - newcols

def todd_iter(m, quiet=True):
    m = m.transpose()
    remove_trivial_cols(m)
    m = m.transpose()
    while True:
        m, reduced = do_todd_single(m)
        if not reduced:
            return m
        if not quiet: print(reduced, end='.')


def todd_simp(gates, qubits):
    phase_poly, parity_polys = phase_gates_to_poly(gates, qubits)
    #print(phase_poly)
    #print(parity_polys)
    m = phase_poly.to_par_matrix()
    m2 = todd_iter(m)

    newgates = []
    parities = []
    for col in m2.transpose().data:
        if sum(col) == 1:
            newgates.append(T(next(i for i in range(qubits) if col[i])))
        else:
            parities.append(col)

    p = MultiLinearPoly()
    p.add_par_matrix(m,False)
    p.add_par_matrix(m2,True)
    newgates.extend(p.to_clifford())

    cnots = parity_network(qubits, parities)
    m = Mat2.id(qubits)
    for cnot in cnots:
        m.row_add(cnot.control, cnot.target)
    data = []
    for p in parity_polys:
        l = [int(i in p.par) for i in range(qubits)]
        data.append(l)
    target_matrix = Mat2(data) * m.inverse()
    gates = target_matrix.to_cnots(optimize=True)
    for gate in reversed(gates):
        cnots.append(CNOT(gate.target,gate.control))

    m = Mat2.id(qubits)
    for i, cnot in enumerate(cnots):
        newgates.append(cnot)
        m.row_add(cnot.control, cnot.target)
        for par in parities:
            if par in m.data: # The parity checks out, so put a phase here
                newgates.append(T(m.data.index(par)))
                parities.remove(par)
                break

    if parities:
        raise ValueError("Still phases left on the stack")

    return newgates