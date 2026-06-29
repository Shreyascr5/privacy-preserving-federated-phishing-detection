from __future__ import annotations

import ipaddress
import math
import re
import socket
from collections import Counter
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit

import numpy as np

SUSPICIOUS_KEYWORDS = {
    "account", "auth", "banking", "confirm", "credential", "invoice", "login",
    "password", "paypal", "recover", "secure", "signin", "support", "update",
    "verify", "wallet", "webscr",
}
SUSPICIOUS_TLDS = {
    "buzz", "click", "country", "download", "fit", "gq", "info", "link", "live",
    "ml", "mom", "monster", "rest", "ru", "support", "tk", "top", "work", "xyz",
}
SHORTENERS = {"bit.ly", "cutt.ly", "is.gd", "ow.ly", "rebrand.ly", "t.co", "tinyurl.com"}
REDIRECT_KEYS = {"continue", "dest", "destination", "next", "redirect", "return", "url"}


@dataclass(frozen=True)
class FeatureResult:
    normalized_url: str
    host: str
    supplied_ip: str
    url_length: int
    has_ip_in_url: bool
    has_at_symbol: bool
    has_hyphen: bool
    suspicious_keyword_count: int
    subdomain_count: int
    uses_https: bool
    redirect_pattern_count: int
    suspicious_tld: bool
    entropy: float
    complexity_score: float
    url_shortener: bool
    punycode: bool
    digit_ratio: float
    ip_valid: bool
    ip_public: bool
    ip_private: bool
    ip_reserved: bool
    suspicious_ip_format: bool
    host_matches_ip: bool
    reverse_dns: str | None = None

    def model_vector(self) -> np.ndarray:
        values = [
            min(self.url_length / 200.0, 1.0),
            float(self.has_ip_in_url), float(self.has_at_symbol), float(self.has_hyphen),
            min(self.suspicious_keyword_count / 4.0, 1.0),
            min(self.subdomain_count / 5.0, 1.0), 1.0 - float(self.uses_https),
            min(self.redirect_pattern_count / 3.0, 1.0), float(self.suspicious_tld),
            min(self.entropy / 6.0, 1.0), self.complexity_score / 100.0,
            float(self.url_shortener), float(self.punycode), min(self.digit_ratio * 3.0, 1.0),
            1.0 - float(self.ip_valid), 1.0 - float(self.ip_public),
            float(self.ip_private or self.ip_reserved), float(self.suspicious_ip_format),
            1.0 - float(self.host_matches_ip),
        ]
        return np.asarray(values, dtype=np.float64)

    def to_dict(self) -> dict:
        return asdict(self)


FEATURE_NAMES = [
    "URL length", "IP embedded in URL", "@ symbol", "Hyphenated host",
    "Suspicious keywords", "Subdomain depth", "Missing HTTPS", "Redirect patterns",
    "Suspicious TLD", "Character entropy", "URL complexity", "URL shortener",
    "Punycode host", "Digit density", "Invalid supplied IP", "Non-public IP",
    "Private/reserved IP", "Suspicious IP formatting", "Domain/IP mismatch",
]


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    return -sum((count / len(text)) * math.log2(count / len(text)) for count in counts.values())


def _parse_url(raw_url: str):
    value = raw_url.strip()
    if not value:
        raise ValueError("Enter a website URL.")
    if "://" not in value:
        value = "https://" + value
    parsed = urlsplit(value)
    if not parsed.hostname:
        raise ValueError("The URL must include a valid hostname.")
    return value, parsed


def _ip_properties(raw_ip: str):
    supplied = raw_ip.strip()
    suspicious_format = bool(re.search(r"[^0-9a-fA-F:.]", supplied)) or supplied.count(".") > 3
    try:
        ip = ipaddress.ip_address(supplied)
        public = ip.is_global
        return ip, True, public, ip.is_private, (
            ip.is_reserved or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified
        ), suspicious_format
    except ValueError:
        return None, False, False, False, False, True


def extract_features(url: str, supplied_ip: str, *, network_checks: bool = False) -> FeatureResult:
    normalized, parsed = _parse_url(url)
    host = parsed.hostname.lower().rstrip(".")
    ip_obj, valid, public, private, reserved, odd_ip = _ip_properties(supplied_ip)
    try:
        ipaddress.ip_address(host)
        host_is_ip = True
    except ValueError:
        host_is_ip = False

    labels = host.split(".")
    subdomains = max(0, len(labels) - 2) if not host_is_ip else 0
    text = (host + parsed.path + "?" + parsed.query).lower()
    keyword_count = sum(1 for keyword in SUSPICIOUS_KEYWORDS if keyword in text)
    redirect_count = sum(1 for key in REDIRECT_KEYS if re.search(rf"(?:[?&]|/){key}(?:=|/)", text))
    tld = labels[-1] if len(labels) > 1 else ""
    digit_ratio = sum(char.isdigit() for char in normalized) / max(len(normalized), 1)
    entropy = _entropy(text)

    complexity_points = (
        min(len(normalized) / 2.5, 30) + min(subdomains * 8, 24)
        + min(keyword_count * 8, 24) + min(normalized.count("%") * 4, 12)
        + min((normalized.count("?") + normalized.count("&") + normalized.count("=")) * 2, 10)
    )
    complexity = min(100.0, complexity_points)

    host_matches_ip = bool(host_is_ip and valid and str(ip_obj) == host)
    reverse_dns = None
    if network_checks and valid:
        try:
            reverse_dns = socket.gethostbyaddr(str(ip_obj))[0]
        except (OSError, socket.herror, socket.gaierror):
            reverse_dns = None
        if not host_is_ip:
            try:
                resolved = {item[4][0] for item in socket.getaddrinfo(host, None)}
                host_matches_ip = str(ip_obj) in resolved
            except (OSError, socket.gaierror):
                host_matches_ip = False

    return FeatureResult(
        normalized_url=normalized, host=host, supplied_ip=supplied_ip.strip(),
        url_length=len(normalized), has_ip_in_url=host_is_ip, has_at_symbol="@" in normalized,
        has_hyphen="-" in host, suspicious_keyword_count=keyword_count,
        subdomain_count=subdomains, uses_https=parsed.scheme.lower() == "https",
        redirect_pattern_count=redirect_count, suspicious_tld=tld in SUSPICIOUS_TLDS,
        entropy=entropy, complexity_score=complexity, url_shortener=host in SHORTENERS,
        punycode="xn--" in host, digit_ratio=digit_ratio, ip_valid=valid, ip_public=public,
        ip_private=private, ip_reserved=reserved, suspicious_ip_format=odd_ip,
        host_matches_ip=host_matches_ip, reverse_dns=reverse_dns,
    )
