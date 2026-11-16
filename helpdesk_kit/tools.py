"""Tool calling: two ticket tools, and a loop that runs them for Claude."""

from __future__ import annotations

import json
from typing import Callable

from .llm import DEFAULT_MODEL, Meter, chat
from .tickets import TicketStore
from .tracing import span

TOOLS = [
    {
        "name": "lookup_ticket",
        "description": (
            "Get one help-desk ticket by its number, including status and priority. Use this "
            "whenever the user mentions a ticket number. Ticket numbers are integers, such as 1042."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticket_id": {"type": "integer", "description": "The ticket number."}},
            "required": ["ticket_id"],
        },
    },
    {
        "name": "create_ticket",
        "description": (
            "Open a new help-desk ticket. Use this only when the user asks for a ticket or has a "
            "problem that needs an engineer. Don't use it for general questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "A short summary, under 10 words."},
                "description": {"type": "string", "description": "What happened, in the user's words."},
                "category": {
                    "type": "string",
                    "enum": ["network", "account_access", "hardware", "software", "email_calendar", "security", "policy"],
                },
                "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
            },
            "required": ["title", "description", "category", "priority"],
        },
    },
]


def run_tool(store: TicketStore, name: str, arguments: dict, requester: str = "demo.user@northgate.example") -> str:
    """Run one tool and return its result as a JSON string."""
    if name == "lookup_ticket":
        ticket = store.lookup_ticket(arguments["ticket_id"])
        return json.dumps(ticket or {"error": f"No ticket with id {arguments['ticket_id']}."})
    if name == "create_ticket":
        return json.dumps(store.create_ticket(requester=requester, **arguments))
    return json.dumps({"error": f"Unknown tool: {name}"})


def tool_loop(
    question: str,
    store: TicketStore,
    system: str = "You are the Northgate IT help-desk assistant. Use your tools. Be brief.",
    model: str = DEFAULT_MODEL,
    max_steps: int = 6,
    on_turn: Callable[[int, object], None] | None = None,
    meter: Meter | None = None,
) -> str:
    """Call the model, run any tools it asks for, send back the results, and repeat."""
    messages = [{"role": "user", "content": question}]
    for step in range(1, max_steps + 1):
        with span("llm", step=step) as current:
            reply = chat(messages, system=system, model=model, tools=TOOLS)
            current.add_usage(reply)
        if meter:
            meter.add(reply)
        if on_turn:
            on_turn(step, reply)
        if reply.stop_reason != "tool_use":
            return reply.text

        messages.append({"role": "assistant", "content": reply.raw.content})
        results = []
        for block in reply.raw.content:
            if block.type == "tool_use":
                with span("tool", tool=block.name, arguments=json.dumps(block.input)) as current:
                    output = run_tool(store, block.name, block.input)
                    current.set(result=output)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})
    return f"Stopped after {max_steps} steps without a final answer."
