import unittest
from unittest import TestCase

from TareaClase1.Perceptron import Perceptron


class TestPerceptron(TestCase):
    def test_eval_perceptron_pass(self):
        weights = [3, 1]
        bias = -3
        perp = Perceptron(weights, bias)
        self.assertEqual(0, perp.eval(1, -1))

    def test_eval_perceptron_fail(self):
        weights = [2, 2]
        bias = -3
        perp = Perceptron(weights, bias)
        self.assertEqual(0, perp.eval(1, 0))

    def test_eval_perceptron_excp(self):
        weights = [1, 1, 1]
        bias = 1
        perp = Perceptron(weights, bias)
        with self.assertRaises(Exception):
            perp.eval(1, 1)


if __name__ == '__main__':
    unittest.main()