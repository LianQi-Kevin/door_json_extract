import threading

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


_thread_local = threading.local()


def get_session() -> requests.Session:
    """获取线程隔离的 requests.Session。"""
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _thread_local.session = session
    return _thread_local.session


def get_with_retry(url: str, **kwargs):
    """封装 GET 请求（带重试）。"""
    return get_session().get(url, **kwargs)


def post_with_retry(url: str, **kwargs):
    """封装 POST 请求（带重试）。"""
    return get_session().post(url, **kwargs)
