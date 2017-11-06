import unittest
from unittest import TestCase

from TareaClase1.Perceptron import ORPerceptron


class TestORPerceptron(TestCase):

    def test_eval_orperceptron_pass(self):
        perp = ORPerceptron()
        self.assertEqual(1, perp.eval(1, 0))

    def test_eval_orperceptron_fail(self):
        perp = ORPerceptron()
        self.assertEqual(0, perp.eval(0, 0))

if __name__ == '__main__':
    unittest.main()
