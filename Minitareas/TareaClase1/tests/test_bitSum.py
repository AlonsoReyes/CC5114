import unittest
from unittest import TestCase

from TareaClase1.Perceptron import BitSum


class TestBitSum(TestCase):

    def test_sum_one(self):
        op = BitSum()
        result, carry = op.sum(1, 0)
        self.assertEqual(1, result)
        self.assertEqual(0, carry)

    def test_sum_zero(self):
        op = BitSum()
        result, carry = op.sum(0, 0)
        self.assertEqual(0, result)
        self.assertEqual(0, carry)

    def test_sum_carry(self):
        op = BitSum()
        result, carry = op.sum(1, 1)
        self.assertEqual(0, result)
        self.assertEqual(1, carry)

if __name__ == '__main__':
    unittest.main()
