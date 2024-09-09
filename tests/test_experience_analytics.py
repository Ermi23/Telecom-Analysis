# import unittest
# import pandas as pd
# from scripts.Experience_Analytics import TelecomExperienceAnalytics

# class TestExperienceAnalyzer(unittest.TestCase):

#     def setUp(self):
#         # Create a sample DataFrame for testing
#         self.data = pd.DataFrame({
#             'MSISDN/Number': ['123', '456', '789'],
#             'Total Traffic': [1000, 1500, 800],
#             'Experience Score': [5, 4, 3]
#         })
#         applications = ['MSISDN/Number', 'Total Traffic', 'Experience Score']
#         self.analyzer = TelecomExperienceAnalytics(self.data, applications)

#     def test_analyze_application_usage(self):
#         app_usage = self.analyzer.analyze_application_usage()
#         self.assertIsNotNone(app_usage)

#     def test_user_experience_summary(self):
#         summary = self.analyzer.user_experience_summary()
#         self.assertEqual(len(summary), 3)

#     def test_analyze_user_clusters(self):
#         clusters = self.analyzer.analyze_user_clusters(n_clusters=2)
#         self.assertEqual(len(clusters), 2)

#     def test_plot_top_applications(self):
#         # Plot function won't return anything, so we'll just check for errors
#         try:
#             self.analyzer.plot_top_applications(top_n=2)
#         except Exception as e:
#             self.fail(f"plot_top_applications() raised {type(e).__name__} unexpectedly!")

#     def test_elbow_method(self):
#         # Again, we're just ensuring the method runs without error
#         try:
#             self.analyzer.elbow_method(max_k=3)
#         except Exception as e:
#             self.fail(f"elbow_method() raised {type(e).__name__} unexpectedly!")

# if __name__ == '__main__':
#     unittest.main()
