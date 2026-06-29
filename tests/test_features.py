import unittest

import numpy as np

from phishing_detection.features import FEATURE_NAMES, extract_features


class FeatureExtractionTests(unittest.TestCase):
    def test_extracts_suspicious_url_signals(self):
        result = extract_features("http://user@paypal-secure-login.xyz/verify?redirect=account", "185.43.22.15")
        self.assertTrue(result.has_at_symbol)
        self.assertTrue(result.has_hyphen)
        self.assertTrue(result.suspicious_tld)
        self.assertGreaterEqual(result.suspicious_keyword_count, 2)
        self.assertFalse(result.uses_https)
        self.assertEqual(result.redirect_pattern_count, 1)
        self.assertEqual(len(result.model_vector()), len(FEATURE_NAMES))
        self.assertTrue(np.all((result.model_vector() >= 0) & (result.model_vector() <= 1)))


    def test_ip_classification(self):
        result = extract_features("https://example.com", "127.0.0.1")
        self.assertTrue(result.ip_valid)
        self.assertTrue(result.ip_private)
        self.assertTrue(result.ip_reserved)
        self.assertFalse(result.ip_public)
