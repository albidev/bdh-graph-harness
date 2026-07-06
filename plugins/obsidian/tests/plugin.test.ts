import { jest, describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import BDHSyncPlugin, { getFetch } from '../src/main';

// Mock console methods
const mockConsoleLog = jest.spyOn(console, 'log').mockImplementation(() => {});
const mockConsoleError = jest.spyOn(console, 'error').mockImplementation(() => {});

describe('BDHSyncPlugin', () => {
  let plugin: any;
  let mockApp: any;
  let mockVault: any;
  let mockStatusBar: any;
  let mockFetch: jest.Mock;

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Create mock fetch
    mockFetch = jest.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
    } as Response);
    
    // Replace global fetch
    globalThis.fetch = mockFetch as any;

    // Mock Obsidian App
    mockVault = {
      on: jest.fn(),
      getAbstractFileByPath: jest.fn(),
    };

    mockApp = {
      vault: mockVault,
    };

    // Mock status bar
    mockStatusBar = {
      setText: jest.fn(),
      style: { color: '' },
    };

    // Create plugin instance
    plugin = new BDHSyncPlugin();
    plugin.app = mockApp;
    plugin.statusBar = mockStatusBar;
    plugin.settings = {
      serverUrl: 'http://localhost:8643',
      debounceMs: 100,
      enabled: true,
      logEnabled: true,
    };
  });

  afterEach(() => {
    mockConsoleLog.mockClear();
    mockConsoleError.mockClear();
  });

  describe('handleFileChange', () => {
    it('should sync on file create', async () => {
      const mockFile = {
        path: 'wiki/test-note.md',
        extension: 'md',
      };

      plugin.handleFileChange(mockFile, 'create');

      // Wait for debounce
      await new Promise(resolve => setTimeout(resolve, 150));

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8643/api/node-update',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            event: 'create',
            path: 'wiki/test-note.md',
          }),
        })
      );
    });

    it('should sync on file modify', async () => {
      const mockFile = {
        path: 'wiki/existing-note.md',
        extension: 'md',
      };

      plugin.handleFileChange(mockFile, 'modify');

      await new Promise(resolve => setTimeout(resolve, 150));

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8643/api/node-update',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            event: 'modify',
            path: 'wiki/existing-note.md',
          }),
        })
      );
    });

    it('should sync on file delete', async () => {
      const mockFile = {
        path: 'wiki/deleted-note.md',
        extension: 'md',
      };

      plugin.handleFileChange(mockFile, 'delete');

      await new Promise(resolve => setTimeout(resolve, 150));

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8643/api/node-update',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            event: 'delete',
            path: 'wiki/deleted-note.md',
          }),
        })
      );
    });

    it('should ignore non-markdown files', async () => {
      const mockFile = {
        path: 'images/screenshot.png',
        extension: 'png',
      };

      plugin.handleFileChange(mockFile, 'create');

      await new Promise(resolve => setTimeout(resolve, 150));

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it('should ignore .obsidian directory', async () => {
      const mockFile = {
        path: '.obsidian/workspace.json',
        extension: 'json',
      };

      plugin.handleFileChange(mockFile, 'create');

      await new Promise(resolve => setTimeout(resolve, 150));

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it('should debounce rapid changes', async () => {
      const mockFile = {
        path: 'wiki/rapid-note.md',
        extension: 'md',
      };

      // Simulate rapid changes
      plugin.handleFileChange(mockFile, 'modify');
      plugin.handleFileChange(mockFile, 'modify');
      plugin.handleFileChange(mockFile, 'modify');

      await new Promise(resolve => setTimeout(resolve, 150));

      // Should only sync once
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    it('should not sync when disabled', async () => {
      plugin.settings.enabled = false;

      const mockFile = {
        path: 'wiki/test-note.md',
        extension: 'md',
      };

      plugin.handleFileChange(mockFile, 'create');

      await new Promise(resolve => setTimeout(resolve, 150));

      expect(mockFetch).not.toHaveBeenCalled();
    });
  });

  describe('handleFileRename', () => {
    it('should sync delete and create on rename', async () => {
      const mockFile = {
        path: 'wiki/renamed-note.md',
        extension: 'md',
      };

      plugin.handleFileRename(mockFile, 'wiki/old-name.md');

      await new Promise(resolve => setTimeout(resolve, 250));

      expect(mockFetch).toHaveBeenCalledTimes(2);
      expect(mockFetch).toHaveBeenNthCalledWith(
        1,
        'http://localhost:8643/api/node-update',
        expect.objectContaining({
          body: JSON.stringify({
            event: 'delete',
            path: 'wiki/old-name.md',
          }),
        })
      );
      expect(mockFetch).toHaveBeenNthCalledWith(
        2,
        'http://localhost:8643/api/node-update',
        expect.objectContaining({
          body: JSON.stringify({
            event: 'create',
            path: 'wiki/renamed-note.md',
          }),
        })
      );
    });
  });

  describe('syncToServer', () => {
    it('should update status bar on success', async () => {
      await plugin.syncToServer('create', 'wiki/test.md');

      expect(mockStatusBar.setText).toHaveBeenCalledWith('BDH ●');
      expect(mockStatusBar.style.color).toBe('var(--text-success)');
    });

    it('should update status bar on error', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      await plugin.syncToServer('create', 'wiki/test.md');

      expect(mockStatusBar.setText).toHaveBeenCalledWith('BDH ✗');
      expect(mockStatusBar.style.color).toBe('var(--text-error)');
    });

    it('should update status bar on HTTP error', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
      } as Response);

      await plugin.syncToServer('create', 'wiki/test.md');

      expect(mockStatusBar.setText).toHaveBeenCalledWith('BDH ✗');
      expect(mockStatusBar.style.color).toBe('var(--text-error)');
    });

    it('should skip if synced too recently', async () => {
      plugin.lastSyncTime.set('wiki/test.md', Date.now());

      await plugin.syncToServer('create', 'wiki/test.md');

      expect(mockFetch).not.toHaveBeenCalled();
    });
  });

  describe('settings', () => {
    it('should load default settings', async () => {
      plugin.loadData = jest.fn().mockResolvedValue({});
      await plugin.loadSettings();

      expect(plugin.settings).toEqual({
        serverUrl: 'http://localhost:8643',
        debounceMs: 1000,
        enabled: true,
        logEnabled: true,
      });
    });

    it('should merge saved settings with defaults', async () => {
      plugin.loadData = jest.fn().mockResolvedValue({
        serverUrl: 'http://custom-server:9999',
      });
      await plugin.loadSettings();

      expect(plugin.settings.serverUrl).toBe('http://custom-server:9999');
      expect(plugin.settings.debounceMs).toBe(1000); // default
    });
  });
});
