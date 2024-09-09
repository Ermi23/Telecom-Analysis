import unittest
import pandas as pd
from scripts.data_analysis import DataAnalysis

class TestDataAnalyzer(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        self.data = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4],
            'C': ['NULL', 'valid', 'N/A', ''],
            'D': [10, None, 30, 40]
        })
        applications = ['A', 'B', 'C', 'D']
        self.analyzer = DataAnalysis(self.data, applications)

    def test_missing_values_summary(self):
        missing_summary = self.analyzer.check_missing_values(print_output=False)
        self.assertEqual(missing_summary['Missing Values'].sum(), 3)

    def test_fill_missing_with_mean(self):
        self.analyzer.fill_missing_with_mean()
        self.assertAlmostEqual(self.analyzer.dataframe['A'][2], 2.33, places=2)
        self.assertAlmostEqual(self.analyzer.dataframe['D'][1], 26.67, places=2)

    # def test_inspect_null_representations(self):
    #     with self.assertLogs() as log:
    #         self.analyzer.inspect_null_representations()
    #         self.assertIn("Column 'C' contains 1 'NULL' values.", log.output[0])
    #         self.assertIn("Column 'C' contains 1 'N/A' values.", log.output[1])
    #         self.assertIn("Column 'C' contains 1 '' values.", log.output[2])

    # def test_aggregate_data(self):
    #     # Assuming aggregate_data returns some form of aggregated output
    #     aggregated_data = self.analyzer.aggregate_all()
    #     self.assertIsNotNone(aggregated_data)

if __name__ == '__main__':
    unittest.main()
