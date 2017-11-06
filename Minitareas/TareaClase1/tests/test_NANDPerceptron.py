import unittest
from unittest import TestCase

from TareaClase1.Perceptron import NANDPerceptron


class TestNANDPerceptron(TestCase):

    def test_eval_nandperceptron_pass(self):
        perp = NANDPerceptron()
        self.assertEqual(1, perp.eval(1, 0))

    def test_eval_nandperceptron_fail(self):
        perp = NANDPerceptron()
        self.assertEqual(0, perp.eval(1, 1))

if __name__ == '__main__':
    unittest.main()
