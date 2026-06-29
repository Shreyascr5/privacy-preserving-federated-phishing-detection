from __future__ import annotations

from .features import FeatureResult


def heuristic_indicators(features: FeatureResult) -> list[dict]:
    checks = [
        (features.url_length > 75, "Long URL", 8, f"URL contains {features.url_length} characters"),
        (features.has_ip_in_url, "IP address embedded in URL", 18, "Hostnames normally use a registered domain"),
        (features.has_at_symbol, "@ symbol in URL", 15, "Text before @ can disguise the true destination"),
        (features.has_hyphen, "Hyphenated domain", 5, "Brand-imitating domains frequently use hyphens"),
        (features.suspicious_keyword_count > 0, "Credential-themed language", min(18, features.suspicious_keyword_count * 6), f"Detected {features.suspicious_keyword_count} suspicious keyword(s)"),
        (features.subdomain_count >= 2, "Deep subdomain chain", min(15, features.subdomain_count * 5), f"Detected {features.subdomain_count} subdomain levels"),
        (not features.uses_https, "HTTPS not used", 12, "Traffic is not protected by TLS"),
        (features.redirect_pattern_count > 0, "Redirect-style parameters", 9, "URL contains navigation or redirect parameters"),
        (features.suspicious_tld, "Higher-risk TLD", 12, "The top-level domain is frequently abused"),
        (features.entropy > 4.3, "High character entropy", 8, f"Entropy is {features.entropy:.2f} bits/character"),
        (features.url_shortener, "URL shortening service", 10, "The final destination is obscured"),
        (features.punycode, "Punycode hostname", 16, "Internationalized text may imitate a trusted brand"),
        (not features.ip_valid, "Invalid IP address", 16, "The supplied IP could not be parsed"),
        (features.ip_valid and not features.ip_public, "Non-public IP address", 8, "The supplied IP is not globally routable"),
        (features.suspicious_ip_format, "Suspicious IP formatting", 10, "The IP input has an unusual representation"),
        (features.ip_valid and not features.host_matches_ip, "Domain/IP consistency unverified", 5, "The supplied IP was not confirmed for this hostname"),
    ]
    return [
        {"name": name, "weight": weight, "detail": detail}
        for active, name, weight, detail in checks if active
    ]


def combined_risk(model_probability: float, indicators: list[dict]) -> tuple[int, str]:
    heuristic = min(100.0, sum(item["weight"] for item in indicators))
    score = round((model_probability * 100 * 0.72) + (heuristic * 0.28))
    level = "High Risk" if score >= 70 else "Medium Risk" if score >= 40 else "Low Risk"
    return int(score), level
