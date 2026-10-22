"""
Static URL Analysis Module for QR Code Detection System.

This module provides purely deterministic, static analysis of decoded QR code URLs.
No network calls, external API lookups, browser automation, or payload execution are performed.
"""

import ipaddress
import re
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, parse_qs, unquote


# Known public URL shortening domains (common in phishing/evasion)
KNOWN_URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "buff.ly",
    "ow.ly", "rebrand.ly", "tiny.cc", "cutt.ly", "qr.net", "shorturl.at",
    "rb.gy", "adf.ly", "bc.vc", "shorte.st", "s.id", "v.gd", "clck.ru"
}

# Suspicious keywords frequently observed in phishing subdomains/paths
SUSPICIOUS_TARGET_KEYWORDS = {
    "login", "signin", "verify", "verification", "secure", "security",
    "account", "banking", "authenticate", "update", "wallet", "recover",
    "password", "credential", "auth", "confirm", "billing", "support"
}

# Dangerous file extensions in URL path
DANGEROUS_EXTENSIONS = {
    ".exe", ".apk", ".bat", ".cmd", ".sh", ".vbs", ".scr", ".pif",
    ".jar", ".msi", ".dll", ".ps1", ".hta", ".zip", ".iso"
}

# Query parameter keys commonly associated with open redirect vectors
OPEN_REDIRECT_PARAM_NAMES = {
    "url", "redirect", "redirect_url", "redirect_to", "next", "dest",
    "destination", "target", "goto", "link", "r", "u", "return_to"
}


class StaticURLAnalyzer:
    """
    Deterministic static URL analyzer evaluating structural risk indicators.
    """

    def __init__(self):
        pass

    def analyze(self, raw_url: str) -> Dict[str, Any]:
        """
        Analyze a decoded URL string and return structural attributes and risk indicators.
        """
        if not isinstance(raw_url, str):
            raw_url = str(raw_url)

        cleaned_url = raw_url.strip()

        # Default structure for malformed/empty URLs
        if not cleaned_url:
            return {
                "url": "",
                "scheme": "",
                "hostname": "",
                "is_https": False,
                "is_ip_address": False,
                "url_length": 0,
                "domain_length": 0,
                "subdomain_count": 0,
                "query_parameter_count": 0,
                "uses_punycode": False,
                "uses_shortener": False,
                "risk_indicators": [
                    {
                        "indicator": "EMPTY_URL",
                        "severity": "CRITICAL",
                        "weight": 50,
                        "description": "Empty or whitespace-only URL string provided."
                    }
                ],
                "risk_score": 50,
                "risk_level": "high",
                "summary": "Malformed or empty URL payload."
            }

        # Check for dangerous non-HTTP schemes prior to full parsing
        lower_url = cleaned_url.lower()
        if lower_url.startswith("javascript:"):
            return self._build_dangerous_scheme_result(cleaned_url, "javascript", "JavaScript execution scheme detected.")
        if lower_url.startswith("data:"):
            return self._build_dangerous_scheme_result(cleaned_url, "data", "Inline data URI scheme detected.")
        if lower_url.startswith("file:"):
            return self._build_dangerous_scheme_result(cleaned_url, "file", "Local filesystem access scheme detected.")

        try:
            parsed = urlparse(cleaned_url)
        except Exception as e:
            return {
                "url": cleaned_url,
                "scheme": "",
                "hostname": "",
                "is_https": False,
                "is_ip_address": False,
                "url_length": len(cleaned_url),
                "domain_length": 0,
                "subdomain_count": 0,
                "query_parameter_count": 0,
                "uses_punycode": False,
                "uses_shortener": False,
                "risk_indicators": [
                    {
                        "indicator": "MALFORMED_URL_SYNTAX",
                        "severity": "CRITICAL",
                        "weight": 50,
                        "description": f"URL failed RFC-3986 parsing: {str(e)}"
                    }
                ],
                "risk_score": 50,
                "risk_level": "high",
                "summary": "Syntactically invalid URL."
            }

        scheme = parsed.scheme.lower()
        netloc = parsed.netloc
        path = parsed.path
        query = parsed.query
        fragment = parsed.fragment

        # If scheme is missing (e.g. "example.com/page"), urlparse puts host in path
        if not scheme and not netloc:
            # Check if it looks like a domain or relative path
            parts = path.split("/", 1)
            candidate_host = parts[0]
            if "." in candidate_host:
                netloc = candidate_host
                path = "/" + parts[1] if len(parts) > 1 else ""

        # Extract userinfo if present in authority (e.g. user:pass@host)
        has_userinfo = "@" in netloc
        clean_hostname = ""
        port = None

        if has_userinfo:
            _, host_part = netloc.split("@", 1)
        else:
            host_part = netloc

        if ":" in host_part:
            # Check IPv6 vs host:port
            if host_part.startswith("[") and "]" in host_part:
                # IPv6 literal
                clean_hostname = host_part[1:host_part.index("]")]
                port_part = host_part[host_part.index("]")+1:]
                if port_part.startswith(":"):
                    port = port_part[1:]
            else:
                clean_hostname = host_part.split(":", 1)[0]
                port = host_part.split(":", 1)[1]
        else:
            clean_hostname = host_part

        clean_hostname = clean_hostname.lower()

        # Check IP address literal
        is_ip = False
        try:
            ipaddress.ip_address(clean_hostname)
            is_ip = True
        except ValueError:
            is_ip = False

        # Subdomain count
        if clean_hostname and not is_ip:
            labels = clean_hostname.split(".")
            # Typically domain.tld has 2 labels, subdomains are above 2
            subdomain_count = max(0, len(labels) - 2)
        else:
            subdomain_count = 0

        # Query parameters count
        try:
            parsed_params = parse_qs(query, keep_blank_values=True)
            query_param_count = len(parsed_params)
        except Exception:
            parsed_params = {}
            query_param_count = 0

        # Checks
        is_https = (scheme == "https")
        uses_punycode = "xn--" in clean_hostname
        uses_shortener = (clean_hostname in KNOWN_URL_SHORTENERS)

        risk_indicators: List[Dict[str, Any]] = []

        # Indicator 1: Scheme validation
        if not scheme:
            risk_indicators.append({
                "indicator": "MISSING_SCHEME",
                "severity": "MODERATE",
                "weight": 20,
                "description": "URL does not specify a protocol scheme (e.g., missing https://)."
            })
        elif scheme == "http":
            risk_indicators.append({
                "indicator": "INSECURE_HTTP_SCHEME",
                "severity": "MODERATE",
                "weight": 15,
                "description": "URL uses unencrypted plaintext HTTP scheme instead of HTTPS."
            })
        elif scheme not in ("http", "https"):
            risk_indicators.append({
                "indicator": "NON_STANDARD_SCHEME",
                "severity": "HIGH",
                "weight": 35,
                "description": f"URL uses unusual protocol scheme '{scheme}'."
            })

        # Indicator 2: Authority with userinfo (@ trick)
        if has_userinfo:
            risk_indicators.append({
                "indicator": "USERINFO_EMBEDDED_IN_AUTHORITY",
                "severity": "CRITICAL",
                "weight": 45,
                "description": "URL contains '@' in authority component, commonly used to disguise destination domain."
            })

        # Indicator 3: IP Address Hostname
        if is_ip:
            risk_indicators.append({
                "indicator": "IP_ADDRESS_HOSTNAME",
                "severity": "HIGH",
                "weight": 35,
                "description": f"Hostname '{clean_hostname}' is an IP literal rather than a registered domain name."
            })

        # Indicator 4: Punycode / IDN
        if uses_punycode:
            risk_indicators.append({
                "indicator": "PUNYCODE_IDN_DOMAIN",
                "severity": "HIGH",
                "weight": 25,
                "description": "Hostname uses Punycode ('xn--') encoding, common in homograph impersonation attacks."
            })

        # Indicator 5: Known URL Shortener
        if uses_shortener:
            risk_indicators.append({
                "indicator": "URL_SHORTENER_SERVICE",
                "severity": "HIGH",
                "weight": 30,
                "description": f"Hostname '{clean_hostname}' is a known public URL shortening service obscuring destination."
            })

        # Indicator 6: Excessive URL Length
        if len(cleaned_url) > 100:
            risk_indicators.append({
                "indicator": "EXCESSIVE_URL_LENGTH",
                "severity": "LOW",
                "weight": 10,
                "description": f"Total URL length ({len(cleaned_url)} characters) exceeds 100 characters."
            })

        # Indicator 7: Excessive Subdomains
        if subdomain_count >= 3:
            risk_indicators.append({
                "indicator": "EXCESSIVE_SUBDOMAINS",
                "severity": "MODERATE",
                "weight": 15,
                "description": f"Hostname contains {subdomain_count} subdomain levels (potential tunneling or brand mimicry)."
            })

        # Indicator 8: Suspicious keywords in subdomains or path
        if clean_hostname:
            host_labels = clean_hostname.split(".")
            # Look at subdomains only (exclude registered base domain)
            subdomain_labels = host_labels[:-2] if len(host_labels) > 2 else []
            for kw in SUSPICIOUS_TARGET_KEYWORDS:
                if any(kw in s for s in subdomain_labels):
                    risk_indicators.append({
                        "indicator": "SUSPICIOUS_KEYWORD_IN_SUBDOMAIN",
                        "severity": "MODERATE",
                        "weight": 20,
                        "description": f"Subdomain contains authentication/security keyword '{kw}'."
                    })
                    break

        # Indicator 9: Multiple hyphens in hostname
        if clean_hostname.count("-") >= 2:
            risk_indicators.append({
                "indicator": "MULTIPLE_HYPHENS_IN_HOSTNAME",
                "severity": "LOW",
                "weight": 10,
                "description": f"Hostname contains {clean_hostname.count('-')} hyphens, common in typosquatting."
            })

        # Indicator 10: Dangerous file extension in path
        lower_path = path.lower()
        for ext in DANGEROUS_EXTENSIONS:
            if lower_path.endswith(ext):
                risk_indicators.append({
                    "indicator": "EXECUTABLE_DOWNLOAD_EXTENSION",
                    "severity": "CRITICAL",
                    "weight": 45,
                    "description": f"URL path ends with potentially dangerous executable extension '{ext}'."
                })
                break

        # Indicator 11: Open redirect query parameters
        for p in parsed_params:
            if p.lower() in OPEN_REDIRECT_PARAM_NAMES:
                val = parsed_params[p][0]
                if val.startswith("http://") or val.startswith("https://"):
                    risk_indicators.append({
                        "indicator": "OPEN_REDIRECT_PARAMETER",
                        "severity": "MODERATE",
                        "weight": 20,
                        "description": f"Query parameter '{p}' contains an embedded external URL destination."
                    })
                    break

        # Indicator 12: Suspicious encoding (hex, null bytes)
        if "%00" in lower_url:
            risk_indicators.append({
                "indicator": "NULL_BYTE_ENCODING",
                "severity": "CRITICAL",
                "weight": 40,
                "description": "URL contains null-byte sequence (%00), often used in path truncation attacks."
            })
        elif cleaned_url.count("%") >= 5:
            risk_indicators.append({
                "indicator": "EXCESSIVE_PERCENT_ENCODING",
                "severity": "LOW",
                "weight": 10,
                "description": f"URL contains {cleaned_url.count('%')} percent-encoded characters."
            })

        # Indicator 13: Path traversal or consecutive slashes
        if "//" in path:
            risk_indicators.append({
                "indicator": "CONSECUTIVE_SLASHES_IN_PATH",
                "severity": "LOW",
                "weight": 5,
                "description": "URL path contains redundant consecutive slashes ('//')."
            })
        if "/../" in path or path.endswith("/.."):
            risk_indicators.append({
                "indicator": "DIRECTORY_TRAVERSAL_IN_PATH",
                "severity": "HIGH",
                "weight": 30,
                "description": "URL path contains directory traversal sequences ('/../')."
            })

        # Indicator 14: Path mimics a domain (e.g. /login.paypal.com/)
        if re.search(r"/[a-z0-9\-]+\.(com|org|net|edu|gov|io)/", lower_path):
            risk_indicators.append({
                "indicator": "DOMAIN_MIMICRY_IN_PATH",
                "severity": "HIGH",
                "weight": 25,
                "description": "URL path embeds a domain name pattern, potentially deceiving visual inspection."
            })

        # Compute total heuristic risk score
        total_risk_score = min(100, sum(ind["weight"] for ind in risk_indicators))

        if total_risk_score == 0:
            risk_level = "none"
            summary = "No obvious static URL risk indicators detected."
        elif total_risk_score < 25:
            risk_level = "low"
            summary = "Low structural risk; minimal non-standard indicators."
        elif total_risk_score < 50:
            risk_level = "moderate"
            summary = "Moderate structural risk; caution advised."
        else:
            risk_level = "high"
            summary = "High structural risk; multiple or critical risk indicators detected."

        return {
            "url": cleaned_url,
            "scheme": scheme,
            "hostname": clean_hostname,
            "is_https": is_https,
            "is_ip_address": is_ip,
            "url_length": len(cleaned_url),
            "domain_length": len(clean_hostname),
            "subdomain_count": subdomain_count,
            "query_parameter_count": query_param_count,
            "uses_punycode": uses_punycode,
            "uses_shortener": uses_shortener,
            "risk_indicators": risk_indicators,
            "risk_score": total_risk_score,
            "risk_level": risk_level,
            "summary": summary
        }

    def _build_dangerous_scheme_result(self, raw_url: str, scheme_name: str, desc: str) -> Dict[str, Any]:
        return {
            "url": raw_url,
            "scheme": scheme_name,
            "hostname": "",
            "is_https": False,
            "is_ip_address": False,
            "url_length": len(raw_url),
            "domain_length": 0,
            "subdomain_count": 0,
            "query_parameter_count": 0,
            "uses_punycode": False,
            "uses_shortener": False,
            "risk_indicators": [
                {
                    "indicator": f"DANGEROUS_{scheme_name.upper()}_SCHEME",
                    "severity": "CRITICAL",
                    "weight": 50,
                    "description": desc
                }
            ],
            "risk_score": 50,
            "risk_level": "high",
            "summary": f"Directly executable or dangerous scheme '{scheme_name}:'."
        }
