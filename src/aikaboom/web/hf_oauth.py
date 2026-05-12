"""Hugging Face OAuth for the HF Spaces deploy.

When a Space declares ``hf_oauth: true`` in its README frontmatter, HF
injects four env vars at runtime:

    OAUTH_CLIENT_ID
    OAUTH_CLIENT_SECRET
    OAUTH_SCOPES         (space-separated, e.g. "openid profile inference-api")
    OPENID_PROVIDER_URL  (issuer base, e.g. https://huggingface.co)

This blueprint implements the standard OAuth2 authorization-code flow
against that issuer. The resulting access token is stashed in the
visitor's Flask session (`session['hf_token']`) and the rest of the app
reads it via :mod:`aikaboom.utils.runtime_creds`.

If the env vars are absent (running locally / outside an HF Space) the
blueprint registers but every route returns 503 so the UI can fall back
to "set HF_TOKEN in env" mode.
"""
from __future__ import annotations

import os
import secrets
from urllib.parse import urlencode

import requests
from flask import Blueprint, jsonify, redirect, request, session, url_for


hf_oauth_bp = Blueprint("hf_oauth", __name__)


def _oauth_configured() -> bool:
    return bool(os.getenv("OAUTH_CLIENT_ID") and os.getenv("OAUTH_CLIENT_SECRET"))


def _issuer() -> str:
    return os.getenv("OPENID_PROVIDER_URL", "https://huggingface.co").rstrip("/")


def _scopes() -> str:
    # ``inference-api`` is the scope required for the Inference Providers
    # router. ``openid profile`` lets us show "Signed in as @user" in the UI.
    return os.getenv("OAUTH_SCOPES", "openid profile inference-api")


@hf_oauth_bp.route("/auth/status")
def status():
    """Lightweight endpoint the UI polls to decide what to show."""
    return jsonify({
        "configured": _oauth_configured(),
        "signed_in": bool(session.get("hf_token")),
        "username": session.get("hf_username"),
    })


@hf_oauth_bp.route("/auth/login")
def login():
    if not _oauth_configured():
        return jsonify({"error": "HF OAuth not configured on this deploy"}), 503
    state = secrets.token_urlsafe(24)
    session["hf_oauth_state"] = state
    params = {
        "client_id": os.environ["OAUTH_CLIENT_ID"],
        "redirect_uri": url_for("hf_oauth.callback", _external=True),
        "response_type": "code",
        "scope": _scopes(),
        "state": state,
    }
    return redirect(f"{_issuer()}/oauth/authorize?{urlencode(params)}")


@hf_oauth_bp.route("/auth/callback")
def callback():
    if not _oauth_configured():
        return jsonify({"error": "HF OAuth not configured"}), 503
    if request.args.get("state") != session.pop("hf_oauth_state", None):
        return jsonify({"error": "OAuth state mismatch"}), 400
    code = request.args.get("code")
    if not code:
        return jsonify({"error": "missing code"}), 400

    token_resp = requests.post(
        f"{_issuer()}/oauth/token",
        data={
            "client_id": os.environ["OAUTH_CLIENT_ID"],
            "client_secret": os.environ["OAUTH_CLIENT_SECRET"],
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": url_for("hf_oauth.callback", _external=True),
        },
        timeout=15,
    )
    if not token_resp.ok:
        return jsonify({"error": "token exchange failed", "detail": token_resp.text}), 502

    payload = token_resp.json()
    access_token = payload.get("access_token")
    if not access_token:
        return jsonify({"error": "no access_token in response"}), 502
    session["hf_token"] = access_token

    # Best-effort: resolve username so the UI can show who's signed in.
    try:
        who = requests.get(
            f"{_issuer()}/oauth/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        if who.ok:
            info = who.json()
            session["hf_username"] = info.get("preferred_username") or info.get("name")
    except Exception:
        pass

    return redirect("/")


@hf_oauth_bp.route("/auth/logout", methods=["POST", "GET"])
def logout():
    session.pop("hf_token", None)
    session.pop("hf_username", None)
    return redirect("/")


__all__ = ["hf_oauth_bp"]
