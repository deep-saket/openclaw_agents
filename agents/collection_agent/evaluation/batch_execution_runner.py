"""Batch execution runner for collection-agent evaluation datasets."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from zipfile import ZIP_DEFLATED, ZipFile

RuntimeFactory = Callable[[Path], Any]


@dataclass(slots=True)
class BatchExecutionRunner:
    """Executes collection-agent conversations from JSONL evaluation datasets."""

    dataset_path: str | Path | None = None
    base_dir: Path | None = None
    output_dir: Path | None = None
    default_user_code: str = "user_a"
    runtime_factory: RuntimeFactory | None = None
    eval_dataset_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.base_dir = (self.base_dir or Path(__file__).resolve().parents[1]).resolve()
        self.eval_dataset_dir = self.base_dir / "eval_dataset"
        self.dataset_path = self._resolve_dataset_path(self.dataset_path)
        self.runtime_factory = self.runtime_factory or self._default_runtime_factory
        self.output_dir = (
            self.output_dir.resolve()
            if isinstance(self.output_dir, Path)
            else (self.base_dir / "runtime" / "eval_runs").resolve()
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def discover_datasets(self) -> list[Path]:
        """Returns JSONL datasets under eval_dataset/."""
        if not self.eval_dataset_dir.exists():
            return []
        return sorted(path for path in self.eval_dataset_dir.glob("*.jsonl") if path.is_file())

    def load_scripts(self) -> list[dict[str, Any]]:
        """Loads one evaluation script per JSONL record."""
        scripts: list[dict[str, Any]] = []
        for line_no, raw in enumerate(self.dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Dataset line {line_no} must be a JSON object.")
            script_id = str(payload.get("script_id", "")).strip()
            if not script_id:
                raise ValueError(f"Dataset line {line_no} is missing required script_id.")
            scripts.append(payload)
        if not scripts:
            raise ValueError(f"Dataset {self.dataset_path} did not contain any script rows.")
        return scripts

    def run(self) -> dict[str, Any]:
        """Executes the dataset and writes artifact files plus a zip bundle."""
        scripts = self.load_scripts()
        run_id = datetime.now(timezone.utc).strftime("batch_%Y%m%dT%H%M%SZ")
        run_dir = self.output_dir / run_id
        traces_dir = run_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        runtime = self.runtime_factory(self.base_dir)
        conversation_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []

        for index, script in enumerate(scripts, start=1):
            script_id = str(script["script_id"]).strip()
            session_id = f"{run_id}-{script_id}"
            turn_result = self._execute_script(
                runtime=runtime,
                session_id=session_id,
                script=script,
            )
            trace_payload = {
                "run_id": run_id,
                "script_id": script_id,
                "session_id": session_id,
                "dataset_path": str(self.dataset_path),
                "script_index": index,
                "dataset_record": script,
                "execution_result": turn_result,
            }
            trace_path = traces_dir / f"{script_id}.json"
            trace_path.write_text(json.dumps(trace_payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")

            conversation_row = {
                "run_id": run_id,
                "script_id": script_id,
                "session_id": session_id,
                "dataset_path": str(self.dataset_path),
                "dataset_record": script,
                "execution_result": turn_result,
            }
            conversation_rows.append(conversation_row)

            summary_rows.append(
                {
                    "run_id": run_id,
                    "script_id": script_id,
                    "session_id": session_id,
                    "dataset_path": str(self.dataset_path),
                    "status": str(turn_result.get("status", "completed")),
                    "expected_response_target": str(script.get("expected_response_target", "")),
                    "final_target": str(turn_result.get("final_target", "")),
                    "customer_turn_count": int(turn_result.get("customer_turn_count", 0)),
                    "hop_count": int(turn_result.get("hop_count", 0)),
                    "trace_path": str(trace_path.relative_to(run_dir)),
                    "error": str(turn_result.get("error", "")),
                }
            )

        conversations_path = run_dir / "conversations.jsonl"
        with conversations_path.open("w", encoding="utf-8") as handle:
            for row in conversation_rows:
                handle.write(json.dumps(row, ensure_ascii=True, default=str))
                handle.write("\n")

        summary_path = run_dir / "run_summary.csv"
        self._write_summary_csv(summary_path, summary_rows)

        validation = self._validate_output_linkage(
            expected_script_ids=[str(item["script_id"]).strip() for item in scripts],
            run_summary_path=summary_path,
            conversations_path=conversations_path,
        )

        zip_path = run_dir.with_suffix(".zip")
        self._write_zip_bundle(zip_path=zip_path, run_dir=run_dir)

        return {
            "run_id": run_id,
            "dataset_path": str(self.dataset_path),
            "dataset_count": len(scripts),
            "run_dir": str(run_dir),
            "zip_path": str(zip_path),
            "discovered_datasets": [str(path) for path in self.discover_datasets()],
            "validation": validation,
        }

    def _resolve_dataset_path(self, dataset_path: str | Path | None) -> Path:
        if dataset_path is not None:
            path = Path(dataset_path)
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
                if not path.exists():
                    candidate = (self.base_dir / dataset_path).resolve()
                    if candidate.exists():
                        path = candidate
            else:
                path = path.resolve()
            if not path.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {path}")
            return path

        default_path = self.eval_dataset_dir / "collection_agent_golden_dataset_v3.jsonl"
        if default_path.exists():
            return default_path

        discovered = self.discover_datasets()
        if discovered:
            return discovered[0]
        raise FileNotFoundError(
            f"No JSONL datasets found under {self.eval_dataset_dir}. "
            "Expected eval_dataset/collection_agent_golden_dataset_v3.jsonl by default."
        )

    def _execute_script(
        self,
        *,
        runtime: Any,
        session_id: str,
        script: dict[str, Any],
    ) -> dict[str, Any]:
        user_code = str(script.get("user_code", self.default_user_code)).strip() or self.default_user_code
        start_result = runtime.start_conversation(
            SimpleNamespace(
                user_code=user_code,
                session_id=session_id,
                soft_cap=int(script.get("soft_cap", 10) or 10),
                hard_cap=int(script.get("hard_cap", 50) or 50),
            )
        )

        customer_turns: list[dict[str, Any]] = []
        for item in script.get("conversation", []):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            if role == "customer":
                customer_turns.append({"text": str(item.get("text", "")).strip()})
                continue
            customer_text = str(item.get("customer", "")).strip()
            if customer_text:
                customer_turns.append({"text": customer_text})

        final_turn: dict[str, Any] = start_result.get("turn", {}) if isinstance(start_result.get("turn"), dict) else {}
        try:
            for turn in customer_turns:
                message = str(turn.get("text", "")).strip()
                if not message:
                    continue
                final_turn = runtime.run_turn(
                    SimpleNamespace(
                        message=message,
                        session_id=session_id,
                        sender="customer",
                        soft_cap=int(script.get("soft_cap", 10) or 10),
                        hard_cap=int(script.get("hard_cap", 50) or 50),
                        timeout_seconds=float(script.get("timeout_seconds", 60.0) or 60.0),
                    )
                )
            status = "completed"
            error = ""
        except Exception as exc:  # pragma: no cover - runtime failure capture
            status = "failed"
            error = str(exc)

        return {
            "status": status,
            "error": error,
            "start_result": start_result,
            "final_turn": final_turn,
            "final_response": str(final_turn.get("final_response", "")),
            "final_target": str(final_turn.get("final_target", "")),
            "hop_count": len(final_turn.get("hops", [])) if isinstance(final_turn.get("hops"), list) else 0,
            "customer_turn_count": len(customer_turns),
        }

    @staticmethod
    def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        fieldnames = [
            "run_id",
            "script_id",
            "session_id",
            "dataset_path",
            "status",
            "expected_response_target",
            "final_target",
            "customer_turn_count",
            "hop_count",
            "trace_path",
            "error",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})

    @staticmethod
    def _validate_output_linkage(
        *,
        expected_script_ids: list[str],
        run_summary_path: Path,
        conversations_path: Path,
    ) -> dict[str, Any]:
        expected = {item.strip() for item in expected_script_ids if item.strip()}
        with run_summary_path.open("r", encoding="utf-8", newline="") as handle:
            summary_ids = {
                str(row.get("script_id", "")).strip()
                for row in csv.DictReader(handle)
                if str(row.get("script_id", "")).strip()
            }
        conversation_ids: set[str] = set()
        for raw in conversations_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            conversation_ids.add(str(payload.get("script_id", "")).strip())

        missing_summary = sorted(expected - summary_ids)
        missing_conversations = sorted(expected - conversation_ids)
        if missing_summary:
            raise ValueError(f"Missing script_id(s) in run_summary.csv: {missing_summary}")
        if missing_conversations:
            raise ValueError(f"Missing script_id(s) in conversations.jsonl: {missing_conversations}")
        return {
            "expected_script_ids": len(expected),
            "run_summary_ids": len(summary_ids),
            "conversation_ids": len(conversation_ids),
            "all_script_ids_in_run_summary": True,
            "all_script_ids_in_conversations": True,
        }

    @staticmethod
    def _write_zip_bundle(*, zip_path: Path, run_dir: Path) -> None:
        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as bundle:
            for path in sorted(run_dir.rglob("*")):
                if path.is_file():
                    bundle.write(path, arcname=str(path.relative_to(run_dir)))

    @staticmethod
    def _default_runtime_factory(base_dir: Path) -> Any:
        from agents.collection_agent.ui.server import CollectionDebugRuntime

        return CollectionDebugRuntime.create(base_dir)
