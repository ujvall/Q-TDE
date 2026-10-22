"""
Unit Tests for StaticURLAnalyzer.

Tests 10 distinct deterministic test cases:
1. Normal clean HTTPS URL
2. Plain HTTP URL
3. IPv4-based URL
4. URL containing userinfo (@ authority trick)
5. Long URL (>100 characters)
6. Multiple subdomains (tunneling/mimicry)
7. Punycode IDN URL (xn--)
8. Shortened URL (known shortener domain)
9. Suspicious encoded URL (null byte / excessive percent)
10. Malformed / dangerous scheme URL (javascript: / empty)
"""

import unittest
from url_analysis.url_analyzer import StaticURLAnalyzer


class TestStaticURLAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = StaticURLAnalyzer()

    def test_1_normal_https_url(self):
        url = "https://example.com/page/42"
        res = self.analyzer.analyze(url)
        self.assertEqual(res["scheme"], "https")
        self.assertTrue(res["is_https"])
        self.assertFalse(res["is_ip_address"])
        self.assertEqual(res["hostname"], "example.com")
        self.assertEqual(res["subdomain_count"], 0)
        self.assertFalse(res["uses_punycode"])
        self.assertFalse(res["uses_shortener"])
        self.assertEqual(res["risk_score"], 0)
        self.assertEqual(res["risk_level"], "none")
        self.assertIn("no obvious static url risk", res["summary"].lower())

    def test_2_http_url(self):
        url = "http://example.com/login"
        res = self.analyzer.analyze(url)
        self.assertEqual(res["scheme"], "http")
        self.assertFalse(res["is_https"])
        indicators = [i["indicator"] for i in res["risk_indicators"]]
        self.assertIn("INSECURE_HTTP_SCHEME", indicators)
        self.assertGreater(res["risk_score"], 0)

    def test_3_ip_based_url(self):
        url = "http://192.168.1.100/admin"
        res = self.analyzer.analyze(url)
        self.assertTrue(res["is_ip_address"])
        self.assertEqual(res["hostname"], "192.168.1.100")
        indicators = [i["indicator"] for i in res["risk_indicators"]]
        self.assertIn("IP_ADDRESS_HOSTNAME", indicators)
        self.assertGreaterEqual(res["risk_score"], 35)

    def test_4_url_containing_userinfo(self):
        url = "https://trusted-bank.com@malicious-phish.ru/account"
        res = self.analyzer.analyze(url)
        self.assertEqual(res["hostname"], "malicious-phish.ru")
        indicators = [i["indicator"] for i in res["risk_indicators"]]
        self.assertIn("USERINFO_EMBEDDED_IN_AUTHORITY", indicators)
        self.assertGreaterEqual(res["risk_score"], 45)

    def test_5_long_url(self):
        long_path = "a" * 120
        url = f"https://example.com/{long_path}"
        res = self.analyzer.analyze(url)
        self.assertGreater(res["url_length"], 100)
        indicators = [i["indicator"] for i in res["risk_indicators"]]
        self.assertIn("EXCESSIVE_URL_LENGTH", indicators)

    def test_6_multiple_subdomains(self):
        url = "https://secure.login.verify.account.phish-target.com/index.html"
        res = self.analyzer.analyze(url)
        self.assertGreaterEqual(res["subdomain_count"], 3)
        indicators = [i["indicator"] for i in res["risk_indicators"]]
        self.assertIn("EXCESSIVE_SUBDOMAINS", indicators)
        self.assertIn("SUSPICIOUS_KEYWORD_IN_SUBDOMAIN", indicators)

    def test_7_punycode_url(self):
        url = "https://xn--pple-43d.com/support"
        res = self.analyzer.analyze(url)
        self.assertTrue(res["uses_punycode"])
        indicators = [i["indicator"] for i in res["risk_indicators"]]
        self.assertIn("PUNYCODE_IDN_DOMAIN", indicators)

    def test_8_shortened_url(self):
        url = "https://bit.ly/3xYzAbC"
        res = self.analyzer.analyze(url)
        self.assertTrue(res["uses_shortener"])
        indicators = [i["indicator"] for i in res["risk_indicators"]]
        self.assertIn("URL_SHORTENER_SERVICE", indicators)

    def test_9_suspicious_encoded_url(self):
        url = "https://example.com/download.php%00.jpg"
        res = self.analyzer.analyze(url)
        indicators = [i["indicator"] for i in res["risk_indicators"]]
        self.assertIn("NULL_BYTE_ENCODING", indicators)
        self.assertGreaterEqual(res["risk_score"], 40)

    def test_10_malformed_and_dangerous_scheme(self):
        # JavaScript scheme
        res_js = self.analyzer.analyze("javascript:alert(document.cookie)")
        self.assertEqual(res_js["scheme"], "javascript")
        indicators_js = [i["indicator"] for i in res_js["risk_indicators"]]
        self.assertIn("DANGEROUS_JAVASCRIPT_SCHEME", indicators_js)
        self.assertEqual(res_js["risk_level"], "high")

        # Empty string
        res_empty = self.analyzer.analyze("")
        self.assertEqual(res_empty["risk_level"], "high")
        indicators_empty = [i["indicator"] for i in res_empty["risk_indicators"]]
        self.assertIn("EMPTY_URL", indicators_empty)


if __name__ == "__main__":
    unittest.main()
