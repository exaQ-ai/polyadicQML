import unittest

import polyadicqml as pqml
import numpy as np


def simple_circ(bdr, x, p):
    bdr.allin(x)

    return bdr


class qmeansTester(unittest.TestCase):
    def setUp(self):
        self.circ = pqml.manyq.mqCircuitML(
            simple_circ, 2, 2
        )

        self.X = np.array(
            [[0., 0.],
             [np.pi/2, np.pi/2],
             [0., np.pi/2]]
        )

        self.model = pqml.QMeans(
            self.circ, 2
        )

        self.model.means = np.array(
            [[0, 0, 0, 1],
             [.25, .25, .25, .25]]
        )

    def test_run_circuit(self):
        out = self.model.run_circuit(self.X)
        self.assertTrue(
            np.allclose(
                out,
                np.array([[0, 0, 0, 1],
                          [.25, .25, .25, .25],
                          [0, 0, 0.5, 0.5]])
            )
        )

    def test_dists(self):
        dists = self.model.dists(self.X[:2])
        # print(dists)
        with self.subTest("Correct class zero distance"):
            self.assertTrue(
                np.allclose(
                    np.diag(dists),
                    0
                )
            )
        with self.subTest("Wrong class high proba"):
            dists += np.diag(np.ones(len(dists)))
            self.assertTrue(
                np.all(dists > 0)
            )

    def test_predict_proba(self):
        probs = self.model.predict_proba(self.X)
        print(probs)
        with self.subTest("All positive"):
            self.assertTrue(
                np.all(probs > 0)
            )

        with self.subTest("Sum to one"):
            self.assertTrue(
                np.allclose(
                    probs.sum(axis=1), 1
                )
            )

    def test_predict(self):
        self.assertTrue(
            np.allclose(
                self.model(self.X), np.array([0, 1, 1])
            )
        )


if __name__ == '__main__':
    unittest.main()
