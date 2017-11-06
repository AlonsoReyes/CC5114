import unittest
from unittest import TestCase

from TareaClase1.Perceptron import NPerceptron


class TestNPerceptron(TestCase):

    def test_eval_nperceptron_pass(self):
        weights = [3, 2, 2]
        bias = -3
        perp = NPerceptron(weights, bias)
        inp = [1, -1, 2]
        self.assertEqual(1, perp.eval(inp))

    def test_eval_nperceptron_fail(self):
        weights = [2, 2, 2]
        bias = -3
        inp = [1, -1, 1]
        perp = NPerceptron(weights, bias)
        self.assertEqual(0, perp.eval(inp))

if __name__ == '__main__':
    unittest.main()
