"""Interactive Aditi demo runner for Collection Agent.

This mirrors the UI "Start User A" initialization, then lets you chat
turn-by-turn in terminal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agents.collection_agent.ui.server import (
    CollectionDebugRuntime,
    RunTurnRequest,
    StartConversationRequest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run interactive Collection Agent session for Aditi/User A.")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Collection agent directory (default: agents/collection_agent).",
    )
    parser.add_argument(
        "--session-id",
        default="collection-user_a-cli",
        help="Session id to use for this interactive run.",
    )
    parser.add_argument("--soft-cap", type=int, default=10, help="Agent loop soft cap per turn.")
    parser.add_argument("--hard-cap", type=int, default=50, help="Agent loop hard cap per turn.")
    parser.add_argument("--timeout-seconds", type=float, default=20.0, help="Per-turn timeout.")
    parser.add_argument(
        "--user-code",
        default="user_a",
        choices=["user_a", "user_b", "user_c"],
        help="Demo user to initialize (default: user_a / Aditi).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    runtime = CollectionDebugRuntime.create(base_dir)

    start_payload = runtime.start_conversation(
        StartConversationRequest(
            user_code=args.user_code,
            session_id=args.session_id,
            soft_cap=max(1, int(args.soft_cap)),
            hard_cap=max(max(1, int(args.soft_cap)), int(args.hard_cap)),
        )
    )
    turn = start_payload.get("turn", {}) if isinstance(start_payload, dict) else {}
    opener = str(turn.get("final_response", "")).strip()

    print(f"Session started: {args.session_id}")
    if opener:
        print(f"agent> {opener}")
    print("Type 'exit' to quit.\n")

    inputs = [
        "Yes this is Aditi.",
        "My phone number is : 9900001001and date of birth is  1991-08-19",
        "Actually, I recently lost my job, so I’m facing some financial problems.",
        "I can give 6000 dollars."
    ]

    i = 0
    while True:
        try:
            if i < len(inputs):
                user_text = inputs[i]
            else:
                user_text = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        result = runtime.run_turn(
            RunTurnRequest(
                message=user_text,
                session_id=args.session_id,
                sender="customer",
                soft_cap=max(1, int(args.soft_cap)),
                hard_cap=max(max(1, int(args.soft_cap)), int(args.hard_cap)),
                timeout_seconds=max(1.0, float(args.timeout_seconds)),
            )
        )
        reply = str(result.get("final_response", "No response generated.")).strip()
        print(f"agent> {reply}")

        hops = result.get("hops", []) if isinstance(result.get("hops"), list) else []
        if hops:
            last_hop = hops[-1] if isinstance(hops[-1], dict) else {}
            node_history = last_hop.get("node_history", [])
            if isinstance(node_history, list) and node_history:
                print(f"[nodes] {' -> '.join(str(n) for n in node_history)}")

        i += 1


if __name__ == "__main__":
    main()

