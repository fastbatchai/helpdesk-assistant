# helpdesk-assistant

The reference app for the FastBatch AI bootcamp: an IT help-desk assistant for **Northgate Logistics**,
a fictional company. All names, tickets, documents and log lines are made up.

The lab notebooks install it in their first cell; you don't need to install it yourself.

```bash
pip install "git+https://github.com/fastbatchai/helpdesk-assistant.git"
```

Contents:
- `helpdesk_kit/`: the assistant's building blocks (`llm`, `tickets`, `retrieval`, `evals`, ...) and its
  data: the knowledge base, tickets, test questions, login log and vendor pages.
- `mcp_server/`: the MCP server used in Module 4.
