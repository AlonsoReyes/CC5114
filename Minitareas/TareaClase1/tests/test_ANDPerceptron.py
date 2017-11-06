import unittest
from unittest import TestCase

from TareaClase1.Perceptron import ANDPerceptron


class TestANDPerceptron(TestCase):

    def test_eval_andperceptron_pass(self):
        perp = ANDPerceptron()
        self.assertEqual(1, perp.eval(1, 1))

    def test_eval_andperceptron_fail(self):
        perp = ANDPerceptron()
        self.assertEqual(0, perp.eval(1, 0))

if __name__ == '__main__':
    unittest.main()