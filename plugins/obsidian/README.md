# BDH Graph Harness Sync

An Obsidian plugin that automatically syncs vault changes to the BDH Graph Harness server.

## Features

- **Automatic sync**: Detects file create, modify, delete, and rename events
- **Debounced updates**: Prevents spamming the server with rapid changes
- **Status bar indicator**: Shows sync status (idle, syncing, ok, error)
- **Configurable**: Adjust server URL, debounce delay, and enable/disable logging
- **Lightweight**: Only triggers on markdown files, ignores `.obsidian` directory

## Installation

### From source

1. Clone or download this repository
2. Run `npm install` to install dependencies
3. Run `npm run build` to build the plugin
4. Copy `manifest.json` and `main.js` to your Obsidian vault's `.obsidian/plugins/bdh-graph-harness-sync/` directory
5. Enable the plugin in Obsidian Settings → Community Plugins

## Configuration

Open Settings → BDH Graph Harness Sync to configure:

- **Server URL**: The URL of your BDH Graph Harness server (default: `http://localhost:8643`)
- **Debounce delay (ms)**: Wait time after last change before syncing (default: 1000ms)
- **Enable sync**: Toggle automatic sync on/off
- **Enable logging**: Toggle console logging of sync events

## How it works

1. The plugin listens to Obsidian's vault events (`create`, `modify`, `delete`, `rename`)
2. When a markdown file changes, it debounces the event (waits for rapid changes to settle)
3. After the debounce period, it sends a POST request to `/api/node-update` on the BDH server
4. The server detects the changes and broadcasts WebSocket events to connected clients
5. The visualization updates in real-time without requiring a full graph rebuild

## Development

### Prerequisites

- Node.js 18+
- npm

### Setup

```bash
cd obsidian-plugin
npm install
```

### Build

```bash
npm run build
```

### Watch mode

```bash
npm run dev
```

### Tests

```bash
npm test
```

## API

The plugin sends POST requests to `{serverUrl}/api/node-update` with:

```json
{
  "event": "create|modify|delete",
  "path": "wiki/note-name.md"
}
```

## License

MIT
