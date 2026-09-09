"""Offline contract tests against the real OpenAI Python SDK request serializer."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from tescogpt.agent.drafting import OpenAIDrafter, draft_schema
from tescogpt.evaluation.judge import OpenAIReplyJudge
from tescogpt.evaluation.reply_review import RATING_DIMENSIONS
from tescogpt.retrieval.outcome import Precedent


def _response_payload(output: dict[str, Any], *, model: str) -> dict[str, Any]:
    return {
        "id": "resp_contract_test",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": "msg_contract_test",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": json.dumps(output),
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 15,
        },
    }


def _mock_client(
    output: dict[str, Any], *, model: str
) -> tuple[OpenAI, list[httpx.Request], httpx.Client]:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_response_payload(output, model=model))

    http_client = httpx.Client(transport=httpx.MockTransport(handle))
    client = OpenAI(
        api_key="contract-test-key",
        base_url="https://openai.invalid/v1",
        http_client=http_client,
    )
    return client, requests, http_client


def _precedent() -> Precedent:
    return Precedent(
        case_id="train-1",
        conversation_id="conversation-1",
        lexical_score=1.0,
        rerank_score=1.0,
        outcome_tier="positive_followup_proxy",
        safety_penalty_flags=(),
        message="Thanks Tesco",
        historical_reply="You're welcome.",
        customer_followup="Thanks.",
    )


def _request_json(requests: list[httpx.Request]) -> dict[str, Any]:
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/responses"
    return json.loads(requests[0].content)


def _walk_schema(value: Any, visit: Callable[[str], None]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            visit(key)
            _walk_schema(child, visit)
    elif isinstance(value, list):
        for child in value:
            _walk_schema(child, visit)


def test_draft_schema_avoids_unsupported_output_constraints() -> None:
    keys: list[str] = []
    _walk_schema(draft_schema(), keys.append)
    assert not {"uniqueItems", "minLength", "maxLength"} & set(keys)


def test_openai_drafter_serializes_a_responses_api_request(tmp_path: Path) -> None:
    output = {
        "predicted_intent": "feedback_praise_or_suggestion",
        "intent_confidence": 0.91,
        "draft_reply": "Thanks for sharing your feedback with Tesco.",
        "used_evidence_case_ids": ["train-1"],
    }
    client, requests, http_client = _mock_client(output, model="resolved-draft-model")
    try:
        drafter = OpenAIDrafter("requested-draft-model", tmp_path / "cache", client)
        candidate = drafter.draft(
            {"case_id": "case-1", "message": "Thanks Tesco", "prior_context": ""},
            [_precedent()],
        )
    finally:
        http_client.close()

    request = _request_json(requests)
    assert request["model"] == "requested-draft-model"
    assert request["store"] is False
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True
    assert candidate.draft_reply == output["draft_reply"]
    assert drafter.provenance()["sdk_version"] == "2.54.0"
    assert drafter.provenance()["resolved_models"] == ["resolved-draft-model"]


def test_openai_judge_serializes_a_responses_api_request(tmp_path: Path) -> None:
    output = {
        **{dimension: 2 for dimension in RATING_DIMENSIONS},
        "critical_error_tags": [],
        "overall_pass": "PASS",
        "rationale": "Safe, relevant, and appropriate.",
    }
    client, requests, http_client = _mock_client(output, model="resolved-judge-model")
    row = {
        "review_id": "review-1",
        "message": "Thanks Tesco",
        "prior_context": "",
        "gold_intent": "feedback_praise_or_suggestion",
        "gold_handling": "AUTO_HANDLE",
        "gold_reason": "NO_ACTION_NEEDED",
        "must_include": "thanks",
        "must_avoid": "invented action",
        "draft_reply": "Thanks for your feedback.",
        "proposed_handling": "AUTO_HANDLE",
        "proposed_reason": "NO_ACTION_NEEDED",
        "evidence_quotes": "[]",
    }
    try:
        judge = OpenAIReplyJudge("requested-judge-model", tmp_path / "cache", client=client)
        rating = judge.rate(row)
    finally:
        http_client.close()

    request = _request_json(requests)
    assert request["model"] == "requested-judge-model"
    assert request["store"] is False
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True
    schema_keys: list[str] = []
    _walk_schema(request["text"]["format"]["schema"], schema_keys.append)
    assert not {"uniqueItems", "minLength", "maxLength"} & set(schema_keys)
    assert rating.overall_pass == "PASS"
    assert judge.provenance()["sdk_version"] == "2.54.0"
    assert judge.provenance()["resolved_models"] == ["resolved-judge-model"]
