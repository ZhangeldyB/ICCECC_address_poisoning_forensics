"""Small JSON-RPC client with rate limiting, retries and batching."""
from __future__ import annotations

import threading
import time

import requests


class RpcError(Exception):
    def __init__(self, code, message):
        super().__init__(f"RPC error {code}: {message}")
        self.code = code
        self.message = str(message)

    @property
    def is_range_problem(self) -> bool:
        m = self.message.lower()
        keys = ("range", "too many", "limit", "exceed", "more than", "response size", "too large", "timeout", "timed out")
        return any(k in m for k in keys)


class Rpc:
    def __init__(self, url: str, max_rps: float = 15, timeout: int = 60, retries: int = 12):
        if not url:
            raise ValueError("RPC URL is empty. Set CHAINSTACK_URL in .env")
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self._interval = 1.0 / max_rps if max_rps else 0.0
        self._lock = threading.Lock()
        self._next = 0.0
        self._local = threading.local()
        self._id = 0

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "s"):
            self._local.s = requests.Session()
        return self._local.s

    def _throttle(self, n: int = 1) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self._interval * n
        if wait > 0:
            time.sleep(wait)

    def _post(self, payload, n: int = 1):
        last = None
        for attempt in range(self.retries):
            self._throttle(n)
            try:
                r = self._session().post(self.url, json=payload, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    last = RuntimeError(f"HTTP {r.status_code}")
                    time.sleep(min(60, 1.5 ** attempt))
                    continue
                if r.status_code == 413:
                    raise RpcError(413, "response size too large")
                if r.status_code in (401, 403):
                    raise PermissionError(
                        f"The RPC provider refused the request (HTTP {r.status_code}): {r.text[:300]}\n"
                        "This usually means the plan does not serve blocks this old (archive data), "
                        "or the key is wrong. Try a more recent window or another provider.")
                r.raise_for_status()
                return r.json()
            except (requests.ConnectionError, requests.Timeout, ValueError) as e:
                last = e
                time.sleep(min(60, 1.5 ** attempt))
        raise RuntimeError(f"RPC request failed after {self.retries} attempts "
                           f"(check your internet connection, then run the command again): {self._redact(last)}")

    def _redact(self, err) -> str:
        """Error text without the endpoint path, which contains the API key."""
        from urllib.parse import urlparse
        text, path = str(err), urlparse(self.url).path
        if len(path) > 1:
            text = text.replace(path, "/<key hidden>")
        return text.replace(self.url, "<rpc url hidden>")

    def call(self, method: str, params: list):
        with self._lock:
            self._id += 1
            rid = self._id
        out = self._post({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        if "error" in out and out["error"]:
            raise RpcError(out["error"].get("code"), out["error"].get("message"))
        return out.get("result")

    def batch(self, method: str, params_list: list[list], size: int = 50) -> list:
        """Order-preserving batch call; failed items come back as None."""
        results: list = []
        for i in range(0, len(params_list), size):
            part = params_list[i:i + size]
            payload = [{"jsonrpc": "2.0", "id": j, "method": method, "params": p} for j, p in enumerate(part)]
            out = self._post(payload, n=len(part))
            if isinstance(out, dict):  # provider rejected batching: fall back to single calls
                for p in part:
                    try:
                        results.append(self.call(method, p))
                    except RpcError:
                        results.append(None)
                continue
            by_id = {o.get("id"): o.get("result") for o in out}
            results.extend(by_id.get(j) for j in range(len(part)))
        return results

    # convenience -----------------------------------------------------
    def block_number(self) -> int:
        return int(self.call("eth_blockNumber", []), 16)

    def block_timestamp(self, number: int) -> int:
        b = self.call("eth_getBlockByNumber", [hex(number), False])
        return int(b["timestamp"], 16)

    def client_version(self) -> str:
        try:
            return str(self.call("web3_clientVersion", []))
        except Exception:
            return "unknown"

    def block_at_or_after(self, ts: int) -> int:
        """First block whose timestamp is >= ts.

        Starts from an estimate (12-second slots) and searches only near it, so the
        node is never asked for very old blocks that need an archive plan.
        """
        latest = self.block_number()
        latest_ts = self.block_timestamp(latest)
        if ts >= latest_ts:
            return latest
        lo = hi = max(1, latest - (latest_ts - ts) // 12)
        step = 2000
        while lo > 1 and self.block_timestamp(lo) >= ts:
            lo, step = max(1, lo - step), step * 2
        step = 2000
        while hi < latest and self.block_timestamp(hi) < ts:
            hi, step = min(latest, hi + step), step * 2
        while lo < hi:
            mid = (lo + hi) // 2
            if self.block_timestamp(mid) < ts:
                lo = mid + 1
            else:
                hi = mid
        return lo