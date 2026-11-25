# ruff: noqa: E501
"""Transactional e-mail through Brevo's REST API (https://developers.brevo.com).

Only three calls are used:
  GET  /v3/account      is the key valid?                 (health check)
  GET  /v3/senders      which sender addresses are verified? (auto-picks one if BREVO_SENDER_EMAIL is unset)
  POST /v3/smtp/email   send one message

Without BREVO_API_KEY the app still works: `LogMailer` prints the message to the server log,
which is enough to try sign-up locally (the verification code shows up in the terminal).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)
BREVO = "https://api.brevo.com/v3"


class Mailer(Protocol):
    def send(self, to_email: str, to_name: str, subject: str, html: str) -> bool: ...


class LogMailer:
    """Fallback when no API key is set: the message goes to the log instead of an inbox."""

    def send(self, to_email: str, to_name: str, subject: str, html: str) -> bool:
        logger.warning("[mail not sent: BREVO_API_KEY unset] to=%s subject=%r\n%s", to_email, subject, html)
        return True


@dataclass
class BrevoMailer:
    api_key: str
    sender_email: str
    sender_name: str

    @classmethod
    def from_settings(cls, settings: Settings) -> BrevoMailer:
        return cls(settings.brevo_api_key, settings.brevo_sender_email, settings.brevo_sender_name)

    def _headers(self) -> dict[str, str]:
        return {"api-key": self.api_key, "accept": "application/json", "content-type": "application/json"}

    def status(self) -> str:
        """'ok' when the key works and a verified sender is known, otherwise a short reason."""
        try:
            with httpx.Client(timeout=10) as client:
                account = client.get(f"{BREVO}/account", headers=self._headers())
                if account.status_code != 200:
                    return f"key rejected ({account.status_code})"
                return "ok" if self.resolve_sender(client) else "no verified sender (set BREVO_SENDER_EMAIL)"
        except httpx.HTTPError as exc:
            return f"unreachable ({type(exc).__name__})"

    def resolve_sender(self, client: httpx.Client) -> str:
        """The address mail is sent from: BREVO_SENDER_EMAIL, or the first verified sender on the account."""
        if self.sender_email:
            return self.sender_email
        response = client.get(f"{BREVO}/senders", headers=self._headers())
        if response.status_code != 200:
            return ""
        senders = [s for s in response.json().get("senders", []) if s.get("active")]
        if senders:
            self.sender_email = senders[0]["email"]
            self.sender_name = self.sender_name or senders[0].get("name", "")
        return self.sender_email

    def send(self, to_email: str, to_name: str, subject: str, html: str) -> bool:
        try:
            with httpx.Client(timeout=15) as client:
                sender = self.resolve_sender(client)
                if not sender:
                    logger.error("Brevo: no verified sender address; set BREVO_SENDER_EMAIL in .env")
                    return False
                response = client.post(
                    f"{BREVO}/smtp/email",
                    headers=self._headers(),
                    json={
                        "sender": {"email": sender, "name": self.sender_name or sender},
                        "to": [{"email": to_email, "name": to_name or to_email}],
                        "subject": subject,
                        "htmlContent": html,
                    },
                )
        except httpx.HTTPError as exc:
            logger.error("Brevo: request failed: %s", exc)
            return False
        if response.status_code >= 300:
            logger.error("Brevo: send failed %s %s", response.status_code, response.text[:300])
            return False
        return True


def get_mailer(settings: Settings) -> Mailer:
    return BrevoMailer.from_settings(settings) if settings.brevo_api_key else LogMailer()


def code_email(name: str, code: str) -> tuple[str, str]:
    return (
        f"{code} is your RAG Chatbot code",
        f"""<div style="font-family:Archivo,Helvetica,Arial,sans-serif;max-width:480px;margin:0 auto;padding:32px 24px;color:#111827">
<p style="font-size:14px;color:#6b7487;margin:0 0 8px">RAG Chatbot</p>
<h1 style="font-size:22px;margin:0 0 16px">Hi {name or "there"}, confirm your e-mail</h1>
<p style="font-size:15px;line-height:1.5">Enter this code on the sign-up page. It expires in 15 minutes.</p>
<p style="font:600 34px/1 'JetBrains Mono',Menlo,monospace;letter-spacing:.3em;margin:24px 0;color:#C2410C">{code}</p>
<p style="font-size:13px;color:#6b7487">If you did not create an account, ignore this message.</p></div>""",
    )


def reset_email(name: str, link: str) -> tuple[str, str]:
    return (
        "Reset your RAG Chatbot password",
        f"""<div style="font-family:Archivo,Helvetica,Arial,sans-serif;max-width:480px;margin:0 auto;padding:32px 24px;color:#111827">
<p style="font-size:14px;color:#6b7487;margin:0 0 8px">RAG Chatbot</p>
<h1 style="font-size:22px;margin:0 0 16px">Hi {name or "there"}, choose a new password</h1>
<p style="font-size:15px;line-height:1.5">This link works once and expires in 30 minutes.</p>
<p style="margin:24px 0"><a href="{link}" style="display:inline-block;background:#C2410C;color:#fff;text-decoration:none;padding:12px 18px;border-radius:8px;font-weight:600">Reset password</a></p>
<p style="font-size:13px;color:#6b7487">Or paste this address into your browser:<br>{link}</p>
<p style="font-size:13px;color:#6b7487">If you did not ask for this, ignore the message. Your password stays as it is.</p></div>""",
    )
