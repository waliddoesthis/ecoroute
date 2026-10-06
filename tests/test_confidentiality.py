"""Confidentiality layer. Fake secrets are assembled at runtime so the repo itself never
contains a string that looks like a live credential."""

from pathlib import Path

import pytest
import yaml

from ecoroute.confidentiality import (
    CallerPolicyLayer,
    Detector,
    Level,
    RulesLayer,
    allowed_models,
    redact,
    restore,
)
from ecoroute.confidentiality.rules import iban_ok, luhn_ok

CATALOG = yaml.safe_load(Path("configs/models.yaml").read_text())["models"]

AWS = "AKIA" + "IOSFODNN7" + "EXAMPLE"
GH = "ghp" + "_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
SLACK = "xox" + "b-" + "1234567890-abcdefghij"
PEM = "-----BEGIN RSA " + "PRIVATE KEY-----\nMIIEow..."
JWT = "eyJ" + "hbGciOiJIUzI1NiJ9" + ".eyJ" + "zdWIiOiIxMjM0NTY3ODkwIn0" + ".dozjgNryP4J3jVmNHl0w5N"

# Prompts that must never reach a model without restricted clearance. Extend this list
# whenever a miss is found; one violation fails the build.
MUST_BE_RESTRICTED = [
    f"Why does boto fail with key {AWS}?",
    f"My token {GH} stopped working",
    f"post to slack with {SLACK}",
    f"Here is the server key:\n{PEM}",
    f"decode this {JWT}",
    "Charge card 4111 1111 1111 1111 for the order",
    "Wire it to GB82 WEST 1234 5698 7654 32 please",
    "Her SSN is 123-45-6789",
    "config: password = hunter2hunter2",
    "connect to postgres://admin:s3cretpw@db.internal.example:5432/prod",
    "DATABASE_URL api_key: 9f8e7d6c5b4a39281706f5e4d3c2b1a0",
]

MUST_BE_CONFIDENTIAL_OR_ABOVE = [
    "Email the report to jane.doe@acme-corp.com",
    "Call me on +33 6 12 34 56 78 tomorrow",
]

PUBLIC_LOOKING = [
    "Translate 'good morning' into Spanish",
    "What is 1234 * 5678?",
    "Explain the difference between TCP and UDP",
    "Write a haiku about autumn leaves",
    "Order number 2024-11-05 shipped in 3 boxes",
    "Event at timestamp 1696543200000 was logged",
]


@pytest.mark.parametrize("prompt", MUST_BE_RESTRICTED)
def test_secrets_and_ids_are_restricted(prompt):
    got = Detector().classify(prompt)
    assert got.level == Level.RESTRICTED, got.reasons()
    allowed = allowed_models(got.level, CATALOG)
    clearance = {m["name"]: Level.parse(m["clearance"]) for m in CATALOG}
    assert allowed and all(clearance[m] == Level.RESTRICTED for m in allowed)


@pytest.mark.parametrize("prompt", MUST_BE_CONFIDENTIAL_OR_ABOVE)
def test_contact_details_are_confidential(prompt):
    assert Detector().classify(prompt).level >= Level.CONFIDENTIAL


@pytest.mark.parametrize("prompt", PUBLIC_LOOKING)
def test_ordinary_prompts_stay_at_the_floor(prompt):
    got = Detector().classify(prompt)
    assert got.level == Level.INTERNAL, got.reasons()
    assert Detector(floor="public").classify(prompt).level == Level.PUBLIC


def test_validators_reject_lookalikes():
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")
    assert iban_ok("GB82WEST12345698765432") and not iban_ok("GB82WEST12345698765433")
    assert Detector(floor="public").classify("card 4111 1111 1111 1112").level < Level.RESTRICTED


def test_caller_label_raises_but_never_lowers():
    d = Detector()
    assert (
        d.classify("hello", {"sensitivity_label": "Highly Confidential"}).level == Level.RESTRICTED
    )
    assert d.classify(f"key {AWS}", {"sensitivity_label": "public"}).level == Level.RESTRICTED
    # Unknown labels are not ignored.
    assert d.classify("hello", {"sensitivity_label": "Board only"}).level == Level.CONFIDENTIAL
    assert d.classify("hello", {"min_level": "confidential"}).level == Level.CONFIDENTIAL


def test_company_dictionary():
    d = Detector([CallerPolicyLayer(), RulesLayer(dictionary={"Project Falcon": "restricted"})])
    assert d.classify("status of project falcon?").level == Level.RESTRICTED
    assert d.classify("a falcon is a bird").level == Level.INTERNAL


def test_reasons_name_the_rule_without_leaking_the_value():
    got = Detector().classify(f"key {AWS}")
    assert any("aws_access_key" in r for r in got.reasons())
    assert all(AWS not in r for r in got.reasons())


def test_redact_and_restore_round_trip():
    text = "Send jane@acme.io and bob@acme.io the file, cc jane@acme.io"
    got = Detector().classify(text)
    masked, mapping = redact(text, got.findings)
    assert "jane@acme.io" not in masked and masked.count("<EMAIL_1>") == 2
    assert Detector().classify(masked).level == Level.INTERNAL
    assert restore(masked, mapping) == text


def test_missing_clearance_means_public_only():
    catalog = [{"name": "a", "clearance": "restricted"}, {"name": "b"}]
    assert allowed_models(Level.INTERNAL, catalog) == ["a"]
    assert allowed_models(Level.PUBLIC, catalog) == ["a", "b"]
