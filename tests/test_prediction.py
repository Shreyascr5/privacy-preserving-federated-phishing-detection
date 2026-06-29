import unittest

from phishing_detection.analyzer import analyze_target


class PredictionTests(unittest.TestCase):
    def test_demo_phishing_case_is_flagged(self):
        report = analyze_target("https://paypal-login-security.xyz/verify/account", "185.43.22.15")
        self.assertEqual(report.prediction, "Potential Phishing Website")
        self.assertGreaterEqual(report.risk_score, 70)
        self.assertEqual(report.model_source, "Federated Global Model")

    def test_demo_legitimate_case_is_not_flagged(self):
        report = analyze_target("https://www.python.org/docs", "8.8.8.8")
        self.assertEqual(report.prediction, "Likely Legitimate Website")
        self.assertLess(report.risk_score, 40)
