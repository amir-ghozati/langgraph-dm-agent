"""Phone validation.

Arguments are validated before anything is committed, and the failure *kind* is
returned rather than a sentence — `phone_corrections_are_distinct` compares
those kinds, so a generic "that doesn't look right" would make the assertion
unfalsifiable and the correction useless to the lead.

Two decisions worth knowing, both found by probing real numbers rather than by
reading the library docs:

* **A national-format number is valid.** `0151 23456789` parses fine with
  `DEFAULT_PHONE_REGION=DE`, because a default region is exactly what a country
  code supplies. Requiring `+49` would reject the way a German lead actually
  types their number. See decision D38.
* **Letters are rejected before parsing.** `phonenumbers` treats them as a
  vanity number and helpfully maps them to digits — `0151 23oh4a78` becomes
  `+4915123644278`, a real, valid, wrong number. In a DM, letters are a typo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

import phonenumbers

LETTERS = re.compile(r"[A-Za-z]")


class PhoneError(StrEnum):
    """The vocabulary `phone_error_names` asserts against."""

    NON_NUMERIC = "non_numeric"
    LANDLINE = "landline"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    UNPARSEABLE = "unparseable"
    MISSING_COUNTRY_CODE = "missing_country_code"
    """Only when no default region is configured. With one, a national-format
    number is valid — see the module docstring."""


ACCEPTED_TYPES = {
    phonenumbers.PhoneNumberType.MOBILE,
    phonenumbers.PhoneNumberType.FIXED_LINE_OR_MOBILE,
}

MESSAGES = {
    PhoneError.NON_NUMERIC: "that one's got some letters in it — could you send just the digits?",
    PhoneError.LANDLINE: "that looks like a landline — do you have a mobile number?",
    PhoneError.TOO_SHORT: "that looks a few digits short — could you double-check it?",
    PhoneError.TOO_LONG: "that's a few digits too long — could you double-check it?",
    PhoneError.UNPARSEABLE: "I couldn't read that as a phone number — could you send it again?",
    PhoneError.MISSING_COUNTRY_CODE: "could you send it with the country code, like +49…?",
}


@dataclass(frozen=True, slots=True)
class PhoneResult:
    ok: bool
    e164: str | None = None
    error: PhoneError | None = None

    @property
    def message(self) -> str | None:
        """What the lead is told. Specific to the actual failure, so a second
        attempt has something to act on."""
        return MESSAGES[self.error] if self.error else None


def validate_phone(raw: str, region: str = "DE") -> PhoneResult:
    text = (raw or "").strip()
    if not text:
        return PhoneResult(False, error=PhoneError.UNPARSEABLE)

    if LETTERS.search(text):
        return PhoneResult(False, error=PhoneError.NON_NUMERIC)

    try:
        number = phonenumbers.parse(text, region)
    except phonenumbers.NumberParseException as exc:
        if exc.error_type == phonenumbers.NumberParseException.TOO_SHORT_NSN:
            return PhoneResult(False, error=PhoneError.TOO_SHORT)
        if exc.error_type == phonenumbers.NumberParseException.INVALID_COUNTRY_CODE:
            return PhoneResult(False, error=PhoneError.MISSING_COUNTRY_CODE)
        return PhoneResult(False, error=PhoneError.UNPARSEABLE)

    if not phonenumbers.is_possible_number(number):
        reason = phonenumbers.is_possible_number_with_reason(number)
        if reason == phonenumbers.ValidationResult.TOO_SHORT:
            return PhoneResult(False, error=PhoneError.TOO_SHORT)
        if reason == phonenumbers.ValidationResult.TOO_LONG:
            return PhoneResult(False, error=PhoneError.TOO_LONG)
        return PhoneResult(False, error=PhoneError.UNPARSEABLE)

    if not phonenumbers.is_valid_number(number):
        return PhoneResult(False, error=PhoneError.UNPARSEABLE)

    if phonenumbers.number_type(number) not in ACCEPTED_TYPES:
        return PhoneResult(False, error=PhoneError.LANDLINE)

    return PhoneResult(
        True, e164=phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)
    )
