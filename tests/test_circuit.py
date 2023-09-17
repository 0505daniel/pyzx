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


import unittest
import random
import sys
import os
if __name__ == '__main__':
    sys.path.append('..')
    sys.path.append('.')

mydir = os.path.dirname(__file__)

try:
    import numpy as np
    from pyzx.tensor import tensorfy, compare_tensors, find_scalar_correction
    import math
except ImportError:
    np = None

try:
    from qiskit import quantum_info
    from qiskit.circuit import QuantumCircuit
    from qiskit.qasm3 import loads
except ImportError:
    QuantumCircuit = None

from pyzx.generate import cliffordT, cliffords
from pyzx.simplify import clifford_simp
from pyzx.extract import extract_circuit
from pyzx.circuit import Circuit

SEED = 1337

@unittest.skipUnless(np, "numpy needs to be installed for this to run")
class TestCircuit(unittest.TestCase):

    def setUp(self):
        c = Circuit(3)
        c.add_gate("CNOT",0,1)
        c.add_gate("S",2)
        c.add_gate("CNOT",2,1)
        self.c = c

    def test_to_graph_and_back(self):
        g = self.c.to_graph()
        c2 = Circuit.from_graph(g)
        self.assertEqual(self.c.qubits, c2.qubits)
        self.assertListEqual(self.c.gates,c2.gates)

    def test_to_qasm_and_back(self):
        s = self.c.to_qasm()
        c2 = Circuit.from_qasm(s)
        self.assertEqual(self.c.qubits, c2.qubits)
        self.assertListEqual(self.c.gates,c2.gates)

    def test_to_qc_and_back(self):
        s = self.c.to_qc()
        c2 = Circuit.from_qc(s)
        self.assertEqual(self.c.qubits, c2.qubits)
        self.assertListEqual(self.c.gates,c2.gates)

    def test_to_quipper_and_back(self):
        s = self.c.to_quipper()
        c2 = Circuit.from_quipper(s)
        self.assertEqual(self.c.qubits, c2.qubits)
        self.assertListEqual(self.c.gates,c2.gates)

    def test_load_quipper_from_file(self):
        c1 = Circuit.from_quipper_file(os.path.join(mydir,"test_circuit.circuit"))
        c2 = Circuit.from_quipper_file(os.path.join(mydir,"test_circuit_nocontrol_noqubits.circuit"))
        self.assertEqual(c1.qubits, c2.qubits)
        self.assertListEqual(c2.gates,c2.gates)

    def test_cliffordT_preserves_graph_semantics(self):
        random.seed(SEED)
        g = cliffordT(4,20,0.2)
        c = Circuit.from_graph(g)
        g2 = c.to_graph()
        t = tensorfy(g,False)
        t2 = tensorfy(g2,False)
        self.assertTrue(compare_tensors(t,t2, False))

    def test_cliffords_preserves_graph_semantics(self):
        random.seed(SEED)
        g = cliffords(5,30)
        c = Circuit.from_graph(g)
        g2 = c.to_graph()
        t = tensorfy(g,False)
        t2 = tensorfy(g2,False)
        self.assertTrue(compare_tensors(t,t2,False))

    def test_circuit_extract_preserves_semantics(self):
        random.seed(SEED)
        g = cliffordT(5, 70, 0.15)
        t = g.to_tensor(False)
        clifford_simp(g, quiet=True)
        c = extract_circuit(g)
        t2 = c.to_tensor(False)
        self.assertTrue(compare_tensors(t,t2,False))

    def test_two_qubit_gate_semantics(self):
        c = Circuit(2)
        c.add_gate("CNOT",0,1)
        cnot_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
        self.assertTrue(compare_tensors(c.to_matrix(),cnot_matrix))
        c = Circuit(2)
        c.add_gate("CZ",0,1)
        cz_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
        self.assertTrue(compare_tensors(c.to_matrix(),cz_matrix))

    def test_verify_equality_permutation_option(self):
        c1 = Circuit(2)
        c2 = Circuit(2)
        c2.add_gate("SWAP",0,1)
        self.assertTrue(c1.verify_equality(c2,up_to_swaps=True))
        self.assertFalse(c1.verify_equality(c2,up_to_swaps=False))

    def test_parser_state_reset(self):
        from pyzx.circuit.qasmparser import QASMParser
        s = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        h q[0];
        """
        p = QASMParser()
        c1 = p.parse(s)
        c2 = p.parse(s)
        self.assertEqual(c2.qubits, 1)
        self.assertEqual(len(c2.gates), 1)
        self.assertTrue(c1.verify_equality(c2))

    def test_parse_qasm3(self):
        qasm3 = Circuit.from_qasm("""
        OPENQASM 3;
        include "stdgates.inc";
        qubit[3] q;
        cx q[0], q[1];
        s q[2];
        cx q[2], q[1];
        """)
        self.assertEqual(self.c.qubits, qasm3.qubits)
        self.assertListEqual(self.c.gates, qasm3.gates)

    def test_qasm3_p_gate(self):
        qasm2 = Circuit.from_qasm("""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rz(pi/2) q[0];
        """)
        qasm3 = Circuit.from_qasm("""
        OPENQASM 3;
        include "stdgates.inc";
        qubit[1] q;
        p(pi/2) q[0];
        """)
        self.assertEqual(qasm2.qubits, qasm3.qubits)
        self.assertListEqual(qasm2.gates, qasm3.gates)

    def test_qasm3_rz_gate(self):
        # `rz` differs by a global phase between OpenQASM 2 and 3.
        qasm2 = Circuit.from_qasm("""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rz(pi/2) q[0];
        """)
        qasm3 = Circuit.from_qasm("""
        OPENQASM 3;
        include "stdgates.inc";
        qubit[1] q;
        rz(pi/2) q[0];
        """)
        t2 = qasm2.to_matrix()
        t3 = qasm3.to_matrix()
        self.assertFalse(compare_tensors(t2, t3, True))
        self.assertTrue(compare_tensors(t2, t3, False))
        sqrt_half = math.sqrt(1/2)
        self.assertAlmostEqual(find_scalar_correction(t2, t3), complex(sqrt_half, sqrt_half))

    @unittest.skipUnless(QuantumCircuit, "qiskit needs to be installed for this test")
    def test_qasm_qiskit_semantics(self):
        """Verify/document qasm gate semantics when imported into pyzx.

        Currently, pyzx's handling of qasm files differs from qiskit, leading
        to user confusion. This unit test documents those differences.
        """

        def compare_gate_matrix_with_qiskit(gates, num_qubits: int, num_angles: int, preserve_scalar):
            for gate in gates:
                for qasm_version in [2, 3]:
                    header = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n" if qasm_version == 2 \
                        else "OPENQASM 3;\ninclude \"stdgates.inc\";\n"
                    params = "" if num_angles == 0 else "({})".format(",".join(["pi/2"] * num_angles))
                    setup = header + f"qreg q[{num_qubits}];\n{gate}{params} "
                    qasm = setup + ", ".join([f"q[{i}]" for i in range(num_qubits)]) + ";\n"
                    c = Circuit.from_qasm(qasm)
                    pyzx_matrix = c.to_matrix()

                    # qiskit uses little-endian ordering
                    qiskit_qasm = setup + ", ".join([f"q[{i}]" for i in reversed(range(num_qubits))]) + ";\n"
                    qc = QuantumCircuit.from_qasm_str(qiskit_qasm) if qasm_version == 2 else loads(qiskit_qasm)
                    qiskit_matrix = quantum_info.Operator(qc).data
                    self.assertTrue(compare_tensors(pyzx_matrix, qiskit_matrix, preserve_scalar),
                        f"Gate: {gate}\nqasm:\n{qasm}\npyzx_matrix:\n{pyzx_matrix}\nqiskit_matrix:\n{qiskit_matrix}")

        # TODO(issue #116): Support all (or at least the most common) OpenQASM 3 gates.
        simple_one_qubit_gates = ['x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg']  # 'y' off by scalar
        one_param_one_qubit_gates = ['u1', 'p', 'rx', 'ry', 'rz']  # 'rx' and 'ry' off by scalar
        two_param_one_qubit_gates = ['u2']  # off by scalar
        three_param_one_qubit_gates = ['u3']  # off by scalar
        simple_two_qubit_gates = ['cx', 'CX', 'cz', 'ch', 'swap']  # 'cy' not supported, 'ch' off by scalar
        one_param_two_qubit_gates = ['crz']  # 'cu1' and 'cp' not supported (issue #153)
        # TODO(issue #102): Fix phases so that all of these tests pass with `preserve_scalar=True`.
        compare_gate_matrix_with_qiskit(simple_one_qubit_gates, 1, 0, False)
        compare_gate_matrix_with_qiskit(one_param_one_qubit_gates, 1, 1, False)
        compare_gate_matrix_with_qiskit(two_param_one_qubit_gates, 1, 2, False)
        compare_gate_matrix_with_qiskit(three_param_one_qubit_gates, 1, 3, False)
        compare_gate_matrix_with_qiskit(simple_two_qubit_gates, 2, 0, False)
        compare_gate_matrix_with_qiskit(one_param_two_qubit_gates, 2, 1, True)

if __name__ == '__main__':
    unittest.main()
