from __future__ import annotations

import csv
import json
from pathlib import Path

from agents.collection_agent.evaluation.batch_execution_runner import BatchExecutionRunner


class _FakeRuntime:
    def __init__(self) -> None:
        self.turns: list[tuple[str, str]] = []

    @classmethod
    def create(cls, base_dir: Path) -> "_FakeRuntime":
        del base_dir
        return cls()

    def start_conversation(self, request) -> dict:
        return {
            "session_id": request.session_id,
            "turn": {
                "final_response": "hello",
                "final_target": "customer",
                "hops": [],
            },
        }

    def run_turn(self, request) -> dict:
        self.turns.append((request.session_id, request.message))
        return {
            "session_id": request.session_id,
            "final_response": f"handled: {request.message}",
            "final_target": "customer",
            "hops": [{"hop": 1}],
        }


def _write_dataset(path: Path) -> None:
    rows = [
        {
            "script_id": "SCRIPT_001",
            "conversation": [
                {"role": "customer", "text": "I can pay now."},
                {"role": "agent", "text": "annotated"},
            ],
            "expected_response_target": "customer",
        },
        {
            "script_id": "SCRIPT_002",
            "conversation": [
                {"role": "customer", "text": "Can I get a settlement?"},
            ],
            "expected_response_target": "discount_planning_agent",
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_batch_execution_runner_defaults_to_eval_dataset_and_discovers_jsonl(tmp_path: Path) -> None:
    base_dir = tmp_path / "collection_agent"
    eval_dir = base_dir / "eval_dataset"
    eval_dir.mkdir(parents=True)
    dataset_path = eval_dir / "collection_agent_golden_dataset_950.jsonl"
    _write_dataset(dataset_path)
    extra = eval_dir / "another_dataset.jsonl"
    extra.write_text('{"script_id":"X","conversation":[]}\n', encoding="utf-8")

    runner = BatchExecutionRunner(base_dir=base_dir, runtime_factory=_FakeRuntime.create)

    assert runner.dataset_path == dataset_path.resolve()
    discovered = runner.discover_datasets()
    assert dataset_path.resolve() in discovered
    assert extra.resolve() in discovered


def test_batch_execution_runner_preserves_script_ids_in_outputs(tmp_path: Path) -> None:
    base_dir = tmp_path / "collection_agent"
    eval_dir = base_dir / "eval_dataset"
    eval_dir.mkdir(parents=True)
    dataset_path = eval_dir / "collection_agent_golden_dataset_950.jsonl"
    _write_dataset(dataset_path)

    runner = BatchExecutionRunner(
        base_dir=base_dir,
        output_dir=base_dir / "runtime" / "eval_runs",
        runtime_factory=_FakeRuntime.create,
    )
    result = runner.run()

    run_dir = Path(result["run_dir"])
    conversations_path = run_dir / "conversations.jsonl"
    summary_path = run_dir / "run_summary.csv"
    zip_path = Path(result["zip_path"])

    assert conversations_path.exists()
    assert summary_path.exists()
    assert zip_path.exists()
    assert result["validation"]["all_script_ids_in_run_summary"] is True
    assert result["validation"]["all_script_ids_in_conversations"] is True

    conversation_ids = set()
    for line in conversations_path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        conversation_ids.add(payload["script_id"])
        assert payload["run_id"] == result["run_id"]
        assert "dataset_record" in payload
        assert payload["dataset_record"]["script_id"] == payload["script_id"]

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary_ids = {row["script_id"] for row in rows}

    assert conversation_ids == {"SCRIPT_001", "SCRIPT_002"}
    assert summary_ids == {"SCRIPT_001", "SCRIPT_002"}
