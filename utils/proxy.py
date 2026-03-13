import os
from urllib.parse import quote

def _is_true(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

def apply_proxy_env():
    enabled = _is_true(os.getenv("PROXY_ENABLED", "false"))
    if not enabled:
        return {
            "enabled": False,
            "proxy_url_with_auth": "",
            "proxy_url_base": "",
            "proxy_auth": None,
        }

    scheme = os.getenv("PROXY_SCHEME", "http").strip() or "http"
    host = os.getenv("PROXY_HOST", "").strip()
    port = os.getenv("PROXY_PORT", "").strip()
    username = os.getenv("PROXY_USERNAME", "").strip()
    password = os.getenv("PROXY_PASSWORD", "")
    no_proxy = os.getenv("NO_PROXY", "localhost,127.0.0.1").strip()

    if not host or not port:
        return {
            "enabled": False,
            "proxy_url_with_auth": "",
            "proxy_url_base": "",
            "proxy_auth": None,
        }

    proxy_url_base = f"{scheme}://{host}:{port}"

    if username and password:
        user_enc = quote(username, safe="")
        pass_enc = quote(password, safe="")
        proxy_url_with_auth = f"{scheme}://{user_enc}:{pass_enc}@{host}:{port}"
        proxy_auth = f"{username}:{password}"
    else:
        proxy_url_with_auth = proxy_url_base
        proxy_auth = None

    os.environ["http_proxy"] = proxy_url_with_auth
    os.environ["https_proxy"] = proxy_url_with_auth
    os.environ["HTTP_PROXY"] = proxy_url_with_auth
    os.environ["HTTPS_PROXY"] = proxy_url_with_auth
    os.environ["no_proxy"] = no_proxy
    os.environ["NO_PROXY"] = no_proxy

    return {
        "enabled": True,
        "proxy_url_with_auth": proxy_url_with_auth,
        "proxy_url_base": proxy_url_base,
        "proxy_auth": proxy_auth,
    }
