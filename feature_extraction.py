import tldextract
import re
from urllib.parse import urlparse
import math

def extract_features_from_url(url):
    """
    Extract features from a URL for phishing detection.
    :param url: The input URL as a string.
    :return: A dictionary of features.
    """
    # Parse the URL
    parsed_url = urlparse(url)
    domain_info = tldextract.extract(url)

    # Feature extraction
    features = {
        'url_length': len(url),
        'number_of_dots_in_url': url.count('.'),
        'number_of_slashes_in_url': url.count('/'),
        'number_of_special_char_in_url': len(re.findall(r'[@_!#$%^&*()<>?/|}{~:]', url)),
        'number_of_digits_in_url': len(re.findall(r'\d', url)),
        'number_of_hyphens_in_url': url.count('-'),
        'is_https': 1 if parsed_url.scheme == 'https' else 0,
        'domain_length': len(domain_info.domain),
        'is_ip_address': 1 if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', domain_info.domain) else 0,
        'entropy_of_url': -sum(
            [(freq / len(url)) * math.log2(freq / len(url)) for freq in map(url.count, set(url)) if freq > 0]
        ),
        # Placeholder for missing features from the training data
        'entropy_of_domain': 0,  # Replace with actual calculation if available
        'average_number_of_dots_in_subdomain': 0,
        'average_number_of_hyphens_in_subdomain': 0,
        'average_subdomain_length': 0,
        'having_anchor': 0,
        # Add other features as needed
    }

    return features
