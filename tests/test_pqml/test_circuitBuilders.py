import unittest

import polyadicqml as pqml
import numpy as np

def make_builder_test(builder_class):

    class BuilderTester(unittest.TestCase):
        def setUp(self):
            self.bdr = builder_class(3, 1)

        def test_empty_circuit(self):
            self.assertIsInstance(self.bdr, pqml.circuitBuilder)

        def test_invalid_idx_raises_in_alldiam(self):
            with self.subTest("out of bounds"):
                idxes = [10, -1, [-1, 5]]
                for idx in idxes:
                    with self.subTest(idx=idx):
                        self.assertRaises(
                            ValueError,
                            self.bdr.alldiam,
                            idx
                        )
            with self.subTest("Wrong type"):
                idxes = [2.1, "a"]
                for idx in idxes:
                    with self.subTest(idx=idx):
                        self.assertRaises(
                            TypeError,
                            self.bdr.alldiam,
                            idx
                        )

        def test_invalid_idx_raises_in_input(self):
            with self.subTest("out of bounds"):
                idxes = [10, -1, [-1, 5]]
                angles = [0, 2.1, np.array([-1, 2])]
                for idx, theta in zip(idxes, angles):
                    with self.subTest(idx=idx, theta=theta):
                        self.assertRaises(
                            ValueError,
                            self.bdr.input,
                            idx, theta
                        )
            with self.subTest("Wrong type"):
                idxes = [2.1, "a"]
                angles = [0, 2.1]
                for idx, theta in zip(idxes, angles):
                    with self.subTest(idx=idx, theta=theta):
                        self.assertRaises(
                            TypeError,
                            self.bdr.input,
                            idx, theta
                        )

        def test_invalid_idx_raises_in_cz(self):
            with self.subTest("out of bounds"):
                a_s = [10, -1, 1, 0, 11, -5]
                b_s = [0, 2, -1, 7, 13, -1]
                for a, b in zip(a_s, b_s):
                    with self.subTest(a=a, b=b):
                        self.assertRaises(
                            ValueError,
                            self.bdr.cz,
                            a, b
                        )
            with self.subTest("Wrong type"):
                a_s = [1.1, 1, 1]
                b_s = [0, 2.5, "a"]
                for a, b in zip(a_s, b_s):
                    with self.subTest(a=a, b=b):
                        self.assertRaises(
                            TypeError,
                            self.bdr.cz,
                            a, b
                        )

    return BuilderTester


class TestQKBuilder(make_builder_test(pqml.qiskit.qkBuilder)):
    pass


class TestQKParallelBuilder(make_builder_test(pqml.qiskit.qkParallelBuilder)):
    pass


class TestIBMQNativeBuilder(make_builder_test(pqml.qiskit.ibmqNativeBuilder)):
    pass


class TestMQBuilder(make_builder_test(pqml.manyq.mqBuilder)):
    def test_input(self):
        circuit = ""
        for i in range(0, 3):
            with self.subTest(i=i):
                self.bdr.input(i, 1.5)
                circuit += f"SX({i})RZ({i},1.5)SX({i})"
                self.assertEqual(
                    str(self.bdr.circuit()),
                    circuit
                )

    def test_allin(self):
        angles = [1.5, 2.3, -1]
        self.bdr.allin(angles)
        bdr2 = pqml.manyq.mqBuilder(3, 10)

        for idx, theta in enumerate(angles):
            bdr2.input(idx, theta)

        self.assertEqual(
            repr(self.bdr.circuit()),
            repr(bdr2.circuit())
        )

    def test_alldiam(self):
        self.bdr.alldiam()
        self.assertEqual(
            str(self.bdr.circuit()),
            "SX(0)SX(1)SX(2)"
        )

    def test_cz(self):
        self.bdr.cz(1, 2)
        self.assertEqual(
            str(self.bdr.circuit()),
            "CZ(1,2)"
        )

if __name__ == '__main__':
    unittest.main()