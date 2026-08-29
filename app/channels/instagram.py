"""Instagram channel — deliberately unimplemented.

Out of scope for this port. The stub exists so the shape of what it would take
is on the record rather than hand-waved, and so `Channel` has a second
non-trivial implementation to be honest about.

What a real implementation needs, beyond the two Protocol methods:

* **Graph API v23.0 messaging**, `POST /{page_id}/messages`, with a Page access
  token obtained through Facebook Login for Business. Instagram Professional
  accounts only.
* **Webhook verification**: the `hub.challenge` handshake on subscribe, and an
  `X-Hub-Signature-256` HMAC check on every delivery. The source workflow
  verifies neither, so anything that can reach its webhook URL can drive it.
* **At-least-once delivery.** Meta retries on non-2xx and can redeliver on
  success. Dedup is on `(channel, channel_message_id)`; the source has none, so
  a redelivered webhook there produces a second reply to the lead.
* **The 24-hour messaging window.** Outside it, only tagged message types are
  permitted. A follow-up scheduled beyond 24h silently fails to send, which is
  the failure a re-engagement sequence hits first.
* **Burst arrival.** Each message arrives as its own webhook, so
  `BURST_WINDOW_SECONDS` debouncing is load-bearing here rather than a nicety:
  without it a three-message burst produces three replies.
* **Read-back for delivery reconciliation.** The Graph API exposes
  `/conversations` and `/messages`, so an implementation could confirm whether
  a send landed and reconcile a crash between "sent" and "recorded". This port
  prefers a missed message to a duplicate precisely because it has not
  implemented that; see decision D14.
* **Rate limits** per page, and **media attachments**, which arrive with no
  text at all and would reach `normalize` as an empty message.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.channels.base import DeliveryReceipt, InboundMessage, OutboundMessage

_MESSAGE = (
    "InstagramChannel is a documented stub. This port ships the CLI channel only; "
    "see the module docstring for the Graph API contract a real implementation "
    "would have to satisfy."
)


class InstagramChannel:
    name = "instagram"

    def receive(self) -> AsyncIterator[InboundMessage]:
        raise NotImplementedError(_MESSAGE)

    async def send(self, message: OutboundMessage) -> DeliveryReceipt:
        raise NotImplementedError(_MESSAGE)
