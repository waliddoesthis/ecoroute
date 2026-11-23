import json

import numpy as np
from fastapi.testclient import TestClient

from ecoroute.adapt import adapt, read_records
from ecoroute.gateway import EchoBackend, create_app
from ecoroute.gateway.logs import DecisionLog, read_logged_outcomes
from ecoroute.predictors import ModelMeanPredictor
from ecoroute.service import EcoRoute

CATALOG = [
    {"name": "small", "provider": "p", "api_id": "s", "clearance": "internal",
     "price_out_per_mtok": 1.0, "anchor": {"model": "a"}},
    {"name": "big", "provider": "p", "api_id": "b", "clearance": "internal",
     "price_out_per_mtok": 20.0, "anchor": {"model": "b"}},
]  # fmt: skip


def topic_embed(texts):
    # Two topics: "sql ..." prompts point one way, everything else the other.
    return np.array([[1.0, 0.0] if t.startswith("sql") else [0.0, 1.0] for t in texts], np.float32)


def make_eco():
    base = ModelMeanPredictor().fit(np.zeros((2, 2)), np.array([[0.7, 0.9]] * 2), ["a", "b"])
    eco = EcoRoute(base, "fake", CATALOG)
    eco.embed = topic_embed
    return eco


def records():
    # On this user's work the small model always fails SQL and always handles the rest.
    out = []
    for i in range(200):
        topic = "sql" if i % 2 else "chat"
        out.append({"prompt": f"{topic} question {i}", "model": "small",
                    "success": 0.0 if topic == "sql" else 1.0})  # fmt: skip
    return out


def test_adapt_learns_users_work_and_checks_on_held_out_prompts(tmp_path):
    eco = make_eco()
    before_sql = eco.route("sql join two tables").model
    report = adapt(eco, records())
    assert report["test"]["helps"]
    assert report["test"]["brier_adapted"] < 0.05 < report["test"]["brier_prior"]
    assert "small" in report["shifts"]
    sql, chat = eco.route("sql join two tables"), eco.route("chat say hello")
    assert before_sql == "small" and sql.model == "big" and chat.model == "small"
    assert any(s.startswith("Your data: small") for s in chat.steps)
    # Saved and loaded with the adaptation.
    eco.save(tmp_path / "r.pt")
    import yaml

    (tmp_path / "models.yaml").write_text(yaml.safe_dump({"models": CATALOG}))
    loaded = EcoRoute.load(tmp_path / "r.pt", catalog=tmp_path / "models.yaml")
    loaded.embed = topic_embed
    assert loaded.route("sql join two tables").model == "big"


def test_read_records_csv_and_jsonl(tmp_path):
    (tmp_path / "a.csv").write_text("prompt,model,success\nhi,small,yes\nho,big,0.5\n")
    (tmp_path / "b.jsonl").write_text(json.dumps({"prompt": "x", "model": "big", "success": False}))
    assert [r["success"] for r in read_records(tmp_path / "a.csv")] == [1.0, 0.5]
    assert read_records(tmp_path / "b.jsonl")[0]["success"] == 0.0


def test_gateway_log_and_feedback_round_trip(tmp_path):
    eco = make_eco()
    log = DecisionLog(tmp_path)
    client = TestClient(create_app(eco, EchoBackend(), decision_log=log))
    key = "AKIA" + "IOSFODNN7" + "EXAMPLE"
    r = client.post("/v1/chat/completions", json={"model": "ecoroute/auto",
                    "messages": [{"role": "user", "content": "chat hello"}]})  # fmt: skip
    rid = r.headers["x-ecoroute-request-id"]
    assert client.post("/feedback", json={"id": rid, "success": False}).status_code == 200
    assert client.post("/feedback", json={"id": rid, "success": 3}).status_code == 400
    log.decision(f"my key is {key}", eco.route("chat x"), "chat")
    assert key not in (tmp_path / "decisions.jsonl").read_text()
    assert read_logged_outcomes(tmp_path) == [
        {"prompt": "chat hello", "model": "small", "success": 0.0}
    ]
