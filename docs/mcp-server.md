# MCP Server — BDH Graph Harness

The BDH Graph Harness ships with a [Model Context Protocol](https://modelcontextprotocol.io) server that exposes the Hebbian knowledge graph as tools to any MCP-compatible client.

## What is MCP?

MCP is an open protocol that standardizes how AI applications connect to external tools and data sources. It's supported by:

- **Claude Desktop** (Anthropic)
- **Cursor** (AI code editor)
- **Windsurf** (AI IDE)
- **Continue** (AI coding assistant)
- **Any MCP-compatible client**

Instead of configuring HTTP endpoints and curl commands, you simply add the BDH MCP server to your client's config and the tools become available in chat.

## Tools exposed

| Tool | Description |
|------|-------------|
| `query` | Query the vault graph — returns a grounded LLM response with source citations, activated notes, new concepts (neurogenesis), and Hebbian updates |
| `stats` | Graph statistics — neuron count, synapse count, top hubs, top Hebbian connections |
| `hebbian` | Full Hebbian synaptic state — all learned connections with weights and frequencies |
| `graph` | Full graph structure — all nodes (notes) and edges (wikilinks) as JSON |
| `refresh` | Force rebuild the graph and re-compute all embeddings (after vault changes) |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the vault

Edit `bdh-config.yaml` to point at your Obsidian vault:

```yaml
vault_path: /path/to/your/vault
```

### 3. Start the MCP server

The server runs in two transport modes:

**stdio** (default — for Claude Desktop, Cursor):

```bash
python -m bdh_graph_harness --mcp
```

**HTTP** (for web-based clients):

```bash
python -m bdh_graph_harness --mcp --mcp-transport http --mcp-port 8644
```

## Client configuration

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "bdh-graph-harness": {
      "command": "python3",
      "args": ["-m", "bdh_graph_harness", "--mcp"],
      "cwd": "/path/to/bdh-graph-harness",
      "env": {
        "OPENROUTER_API_KEY": "your-key-here"
      }
    }
  }
}
```

Restart Claude Desktop. You'll see the BDH tools available in the chat.

### Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "bdh-graph-harness": {
      "command": "python3",
      "args": ["-m", "bdh_graph_harness", "--mcp"],
      "cwd": "/path/to/bdh-graph-harness",
      "env": {
        "OPENROUTER_API_KEY": "your-key-here"
      }
    }
  }
}
```

### Windsurf / Continue

Similar JSON config — see your client's MCP documentation. The server command is always:

```
python3 -m bdh_graph_harness --mcp
```

### HTTP transport

For clients that prefer HTTP:

```bash
# Start the server
python -m bdh_graph_harness --mcp --mcp-transport http --mcp-port 8644
```

Then configure your client with the URL: `http://localhost:8644/mcp`

## Architecture

The MCP server is a thin transport layer over the existing `bdh_graph_harness` package. It does **not** depend on the HTTP API server (`--serve`) — it imports the package functions directly:

```
MCP Client (Claude Desktop / Cursor / ...)
    ↓ (stdio or HTTP)
MCP Server (bdh_graph_harness/mcp_server.py)
    ↓ (direct imports)
bdh_graph_harness package
    ├── graph.build_graph
    ├── retrieval.attention
    ├── memory.hebbian_update
    ├── llm.llm_respond
    └── neurogenesis.extract_new_concepts
```

This means the Hebbian architecture is fully preserved — the MCP server is just another way to access the same retrieval pipeline. You can run both `--serve` (HTTP API + visualization) and `--mcp` (MCP server) independently or simultaneously.

## How queries work

When you call the `query` tool:

1. **Attention** — hybrid search (vector + BM25) finds seed notes, k-hop graph traversal spreads activation, adaptive threshold filters noise
2. **Hebbian update** — co-activated notes strengthen their synaptic weight (online plasticity, before LLM)
3. **LLM response** — grounded in activated notes only, with citations as `[from: Note Title]`
4. **Neurogenesis** — LLM extracts new concepts not in the vault, creates notes in `concepts/`

The graph learns from every query — the next query operates on a modified graph with stronger synaptic connections.

## Testing with MCP Inspector

You can test the server with the official MCP Inspector:

```bash
npx @modelcontextprotocol/inspector python3 -m bdh_graph_harness --mcp
```

This opens a web UI where you can call each tool and inspect the results.