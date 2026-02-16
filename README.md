# langgraph-mcp-agent

Developer README (English): overview and quick-start for contributors.

## Summary

LangGraph MCP Agent is an advanced programming agent that integrates Clean Architecture patterns with orchestration based on LangGraph and MCP tools (Microservice/Model Control Plane). The agent explores documentation, designs architectures, and generates code using a modular pipeline (LangGraph nodes) and adapters for MCP, LLMs, and vector storage.

This repository contains two main parts:
- The agent (`agent/`): the FastAPI application and agent logic (domain, use cases, adapters, LangGraph graphs).
- The MCP server (`mcp_server/`): an implementation of a server that exposes tools and documentation consumed by the agent.

## Goals

- Provide an architecture reference demonstrating advanced patterns (Result, Unit of Work, Repository, Event Bus, Ports & Adapters).
- Orchestrate software-engineering tasks (research, design, code generation, and testing) with a reusable node graph.
- Integrate with an MCP server for documentation search/queries and tool-assisted workflows.

## Project structure (summary)

- `agent/`
  - `src/`
    - `main.py` - Entry point for the agent application (FastAPI).
    - `api/` - API routes and middleware.
    - `core/`
      - `domain/` - Domain entities, events, errors and patterns (Result, events, etc.).
      - `application/` - Ports and use cases (for example `code_generation.py`).
      - `infrastructure/` - Adapters (MCP, OpenAI, Qdrant), in-memory persistence and LangGraph code (`agent_graph.py`, `nodes.py`, `state.py`).
  - `examples/` - Executable examples such as `architecture_patterns_demo.py` that demonstrate patterns and flows.

- `mcp_server/`
  - `src/server.py` - MCP server implementation and lifecycle management.
  - `src/protocols/` - MCP protocol definitions.
  - `src/tools/` - Tools exposed by the server (documentation, analysis, etc.).
  - `src/docker/` - Dockerfile and docker-compose for local deployment.

- Top-level files:
  - `IMPLEMENTATION_SUMMARY.md` - Summary of architectural and implementation decisions.
  - `README.md` (this file) - General guide.

## Key modules and highlights

- `agent/src/core/application/use_cases/code_generation.py`
  - Central use case: coordinates documentation search, architecture design, and code generation.
  - Uses structured prompts and validation patterns.

- `agent/src/core/infrastructure/adapters/mcp_adapter.py`
  - WebSocket / HTTP client to communicate with the MCP server.
  - Provides methods to list tools, search documentation, analyze code quality, and execute tools.

- `agent/src/core/infrastructure/langgraph/`
  - `agent_graph.py`, `nodes.py`, `state.py` — implement the node graph that runs the agent pipeline (analysis, research, design, generation, and testing).

- `mcp_server/src/server.py`
  - Initializes and exposes the MCP server; includes a `lifespan` manager for startup and shutdown.
  - Hosts tools and endpoints consumed by the `MCPClientAdapter`.

## Requirements & environment

- Python 3.8+ (Python 3.10/3.11 recommended depending on `pyproject.toml` in subpackages).
- A virtual environment (`venv`) or `poetry` is recommended when working with the provided `pyproject.toml` files.

Quick install with venv (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

Install dependencies (from repo root or from each package folder):

```powershell
pip install -r requirements.txt  # if present
```

If the project uses `pyproject.toml`/`poetry`:

```powershell
poetry install
```

Note: The `agent/` and `mcp_server/` folders may contain their own `pyproject.toml` and dependencies.

## Running (development)

- Run the agent (FastAPI) from `agent/src/main.py`:
  - Ensure dependencies are installed in the `agent/` package.
  - Example (adjust path if running from root or from `agent/`):

```powershell
uvicorn agent.src.main:app --reload --port 8000
```

- Run the MCP server:
  - Ensure dependencies are installed in the `mcp_server/` package.
  - From `mcp_server/` you can run `python src/server.py` or use:

```powershell
uvicorn mcp_server.src.server:app --reload --port 9000
```

  - The `server.py` includes a `lifespan` context manager that handles internal component startup/shutdown.

- Run examples:

```powershell
python examples/architecture_patterns_demo.py
```

The demo shows usage of patterns (Result, EventBus, Unit of Work) and in-memory repository examples.

## Development & testing

- No unit test suite was found at the repository root. The `ARCHITECTURE_PATTERNS.md` document contains small validation snippets.
- Recommended: add unit tests for `CodeGenerationUseCase`, the MCP adapters, and `AgentGraph` logic.

## Extensibility

- Adapters: Implement new adapters that satisfy the interfaces in `agent/src/core/application/ports/` to add new LLMs, vector stores, or MCP clients.
- Persistence: Replace `in_memory.py` with real implementations (SQLAlchemy, Redis, etc.) while keeping the `UnitOfWork` and `Repository` interfaces.
- Nodes: Add new LangGraph nodes in `agent/src/core/infrastructure/langgraph/nodes.py` for additional pipeline phases.

## Important files to review

- `agent/src/main.py` — agent startup and configuration.
- `agent/src/core/application/use_cases/code_generation.py` — main use case.
- `agent/src/core/infrastructure/adapters/mcp_adapter.py` — MCP client adapter.
- `agent/src/core/infrastructure/langgraph/*` — graph, nodes, and state.
- `mcp_server/src/server.py` — MCP server and exposed tools.
- `examples/architecture_patterns_demo.py` — runnable demo.
- `IMPLEMENTATION_SUMMARY.md` — detailed implementation notes.

## Suggested next steps

- Add a consolidated `requirements.txt` or document `poetry` usage per package.
- Add a development guide for running the agent and server together (e.g., with `docker-compose` available under `mcp_server/src/docker`).
- Add automated tests and CI.
- Document required environment variables and provide example configuration in `agent/src/core/infrastructure/config/settings.py` if applicable.

## Contributing

1. Open an issue describing the improvement or bug.
2. Create a branch with small, focused changes and tests.
3. Submit a Pull Request with a clear description and links to related issues.

## License

See the `LICENSE` file in this repository for license details.

---

If you want, I can:
- Produce a more detailed README with API examples (endpoints) and usage snippets.
- Create a `CONTRIBUTING.md` and a template `requirements.txt` or consolidated `pyproject.toml`.
- Add Docker / docker-compose instructions to run the agent and server together.

Tell me which of the above you'd like and I'll add it.
