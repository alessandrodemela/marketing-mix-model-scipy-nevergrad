import unittest
import numpy as np
import os
import sys
# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
from data_transformer import DataTransformer


class TestCalculateAdstock(unittest.TestCase):
    def setUp(self):
        """Initialize the DataTransformer before each test"""
        self.transformer = DataTransformer()

    def test_calculate_adstock_float_values(self):
        """Test adstock calculation with float input"""
        input_data = np.array([0.0, 1.0, 0.0, 0.0])
        decay_rate = 0.5
        expected = np.array([0.0, 1.0, 0.5, 0.25])
        result = self.transformer.calculate_adstock(input_data, decay_rate)
        np.testing.assert_array_almost_equal(result, expected)

    def test_calculate_adstock_integer_input(self):
        """Test adstock calculation with integer input -
        should still return float"""
        input_data = np.array([0, 1, 0, 0])  # Integer input
        decay_rate = 0.5
        expected = np.array([0.0, 1.0, 0.5, 0.25])
        result = self.transformer.calculate_adstock(input_data, decay_rate)
        np.testing.assert_array_almost_equal(result, expected)

    def test_calculate_adstock_list_input(self):
        """Test adstock calculation with list input"""
        input_data = [0, 1, 0, 0]  # List input
        decay_rate = 0.5
        expected = np.array([0.0, 1.0, 0.5, 0.25])
        result = self.transformer.calculate_adstock(input_data, decay_rate)
        np.testing.assert_array_almost_equal(result, expected)

    def test_calculate_adstock_decay_rate_bounds(self):
        """Test adstock calculation with boundary decay rates"""
        input_data = np.array([0, 1, 0, 0])

        # Test with decay_rate = 0
        result_zero = self.transformer.calculate_adstock(input_data, 0.0)
        expected_zero = np.array([0.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result_zero, expected_zero)

        # Test with decay_rate = 1
        result_one = self.transformer.calculate_adstock(input_data, 1.0)
        expected_one = np.array([0.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result_one, expected_one)

    def test_calculate_adstock_empty_input(self):
        """Test adstock calculation with empty input"""
        input_data = np.array([])
        decay_rate = 0.5
        result = self.transformer.calculate_adstock(input_data, decay_rate)
        self.assertEqual(len(result), 0)

    def test_calculate_adstock_single_value(self):
        """Test adstock calculation with single value input"""
        input_data = np.array([1.0])
        decay_rate = 0.5
        expected = np.array([1.0])
        result = self.transformer.calculate_adstock(input_data, decay_rate)
        np.testing.assert_array_almost_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
