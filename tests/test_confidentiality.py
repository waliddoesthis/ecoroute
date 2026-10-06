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


def fake_tagger(entities):
    return lambda text: [
        {
            "entity_group": label,
            "score": score,
            "start": text.index(word),
            "end": text.index(word) + len(word),
        }
        for label, word, score in entities
        if word in text
    ]


def test_ner_layer_levels_and_grey_zone():
    from ecoroute.confidentiality import NERLayer

    tagger = fake_tagger(
        [("GIVENNAME", "Amira", 0.98), ("SOCIALNUM", "AB123456", 0.4), ("CITY", "Lyon", 0.1)]
    )
    d = Detector([CallerPolicyLayer(), RulesLayer(), NERLayer(tagger)])
    got = d.classify("Amira from Lyon, ID AB123456")
    kinds = {f.kind: f.level for f in got.findings}
    assert kinds["givenname"] == Level.CONFIDENTIAL
    assert kinds["socialnum"] == Level.CONFIDENTIAL  # uncertain: reported, not dropped
    assert "city" not in kinds  # below the grey zone
    assert got.level == Level.CONFIDENTIAL
    assert d.classify("Amira's id AB123456 again").level == Level.CONFIDENTIAL
    sure = NERLayer(fake_tagger([("SOCIALNUM", "AB123456", 0.9)]))
    assert Detector([sure]).classify("id AB123456").level == Level.RESTRICTED


def test_ner_findings_can_be_redacted():
    from ecoroute.confidentiality import NERLayer

    d = Detector([NERLayer(fake_tagger([("GIVENNAME", "Amira", 0.98)]))])
    masked, mapping = redact(
        "Write a birthday note for Amira", d.classify("Write a birthday note for Amira").findings
    )
    assert masked == "Write a birthday note for <GIVENNAME_1>" and restore(
        masked, mapping
    ).endswith("Amira")


def test_ner_scans_long_text_in_bounded_windows():
    from ecoroute.confidentiality import NERLayer

    seen = []

    def tagger(chunk):
        seen.append(len(chunk))
        i = chunk.find("Amira")
        return (
            []
            if i < 0
            else [{"entity_group": "GIVENNAME", "score": 0.99, "start": i, "end": i + 5}]
        )

    text = "x" * 5000 + " Amira " + "y" * 5000
    layer = NERLayer(tagger, window=1000, overlap=100)
    got = layer.scan(text)
    assert max(seen) <= 1000 and len(got) == 1  # found once despite overlapping windows
    assert text[got[0].start : got[0].end] == "Amira"
    # A name straddling a window edge is whole in the next window.
    edge = "z" * 997 + "Amira" + "z" * 50
    assert [text_f.kind for text_f in layer.scan(edge)] == ["givenname"]


def test_lone_first_name_is_not_confidential_but_a_full_identity_is():
    from ecoroute.confidentiality import NERLayer

    ner = NERLayer(
        fake_tagger(
            [("GIVENNAME", "Tom", 0.99), ("SURNAME", "Okafor", 0.99), ("CITY", "Paris", 0.99)]
        )
    )
    from ecoroute.confidentiality.ner import WEAK_KINDS

    d = Detector([CallerPolicyLayer(), RulesLayer(), ner], weak_kinds=WEAK_KINDS)
    assert d.classify("Tom has 3 apples and eats one").level == Level.INTERNAL
    assert d.classify("cheap flights to Paris?").level == Level.INTERNAL
    assert d.classify("Tom Okafor lives here").level == Level.CONFIDENTIAL
    assert d.classify("email Tom at tom@acme.io").level == Level.CONFIDENTIAL
    # A caller label is not personal content and doesn't make a lone name count.
    got = d.classify("Tom has 3 apples", {"sensitivity_label": "internal"})
    assert got.level == Level.INTERNAL
    assert Detector([ner]).classify("Tom has 3 apples").level == Level.CONFIDENTIAL  # default


def test_slow_layers_are_skipped_once_restricted():
    calls = []

    class Slow:
        name = "slow"

        def scan(self, text, context=None):
            calls.append(text)
            return []

    d = Detector([CallerPolicyLayer(), RulesLayer(), Slow()])
    d.classify(f"key {AWS}")
    assert calls == []  # rules already said restricted
    d.classify("hello")
    assert calls == ["hello"]
    Detector([RulesLayer(), Slow()], stop_at_restricted=False).classify(f"key {AWS}")
    assert len(calls) == 2


def test_scan_many_matches_scan_and_batches_pipeline_calls():
    from ecoroute.confidentiality import NERLayer

    calls = []

    class Pipe:
        tokenizer = object()  # looks like a Hugging Face pipeline

        def __call__(self, chunks, batch_size=1):
            calls.append(len(chunks) if isinstance(chunks, list) else 1)
            one = lambda c: [  # noqa: E731
                {"entity_group": "EMAIL", "score": 0.9, "start": i, "end": i + 5}
                for i in [c.find("a@b.c")]
                if i >= 0
            ]
            return [one(c) for c in chunks] if isinstance(chunks, list) else one(chunks)

    layer = NERLayer(Pipe(), window=100, overlap=10)
    texts = ["mail a@b.c now", "", "x" * 250 + " a@b.c"]
    many = layer.scan_many(texts)
    assert calls == [1 + 3]  # one call for every window of every non-empty text
    assert many[1] == []
    assert [f.start for f in many[2]] == [251]
    assert many == [layer.scan(t) for t in texts]


def test_kind_min_drops_low_scored_weak_kinds_only():
    from ecoroute.confidentiality import NERLayer

    tagger = fake_tagger([("GIVENNAME", "Amira", 0.6), ("SOCIALNUM", "AB123456", 0.6)])
    text = "Amira has AB123456"
    kinds = {f.kind for f in NERLayer(tagger, kind_min={"GIVENNAME": 0.85}).scan(text)}
    assert kinds == {"socialnum"}
    assert {f.kind for f in NERLayer(tagger, kind_min={}).scan(text)} == {"givenname", "socialnum"}
    assert {f.kind for f in NERLayer(tagger).scan(text)} == {"socialnum"}  # default floors
