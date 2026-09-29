import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from data.common.config import Settings


def make_session(s: Settings) -> requests.Session:
    retry = Retry(
        total=s.max_retries,
        backoff_factor=s.backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = s.user_agent
    return session