# echomem-agent

Standalone Agent Playground for EchoMemory.

This project intentionally avoids importing `echomem.*`. It talks to EchoMemory through HTTP only, so it can live in a separate repository.

## Run

```powershell
$env:ECHOMEM_URL="http://127.0.0.1:8000"
$env:PYTHONPATH="src"
python -m agent.server
```

Open http://127.0.0.1:8765.

By default the agent reads Alibaba DashScope-compatible settings from
`E:\KVCache\observer\observer-config.json` and uses the `alibaba` provider entry
plus its mapped model. Set `OBSERVER_CONFIG` to point at another observer config.

The agent page exposes `/agent/chat`, which opens the EchoMemory session, writes
the user message, calls `/api/retrieval/search`, sends the assembled context to a
compatible `/chat/completions` endpoint, and writes the assistant response back
to EchoMemory.
