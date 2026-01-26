"""Alpaca broker adapter (minimal REST)."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import requests


def _base_url() -> str:
    return os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


def _headers() -> Dict[str, str]:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET must be set")
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }


def submit_order(order: Dict[str, Any]) -> Dict[str, Any]:
    url = f\"{_base_url()}/v2/orders\"
    resp = requests.post(url, json=order, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def list_positions() -> List[Dict[str, Any]]:
    url = f\"{_base_url()}/v2/positions\"
    resp = requests.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_account() -> Dict[str, Any]:
    url = f\"{_base_url()}/v2/account\"
    resp = requests.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()
