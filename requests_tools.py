import threading
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

_thread_local = threading.local()


def get_session(proxies: Optional[dict[str, str]] = None, retry_times: int = 3, retry_delay: int = 5) -> requests.Session:
    # 如果当前线程尚未创建 Session，则新建并配置
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        # 全局代理
        if proxies:
            session.proxies.update(proxies)
        # 重试策略
        retry = Retry(
            total=retry_times,
            backoff_factor=retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _thread_local.session = session
    return _thread_local.session


def get_with_retry(url, **kwargs):
    return get_session().get(url, **kwargs)


def post_with_retry(url, **kwargs):
    return get_session().post(url, **kwargs)
