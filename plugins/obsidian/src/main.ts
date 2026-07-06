import { App, Plugin, PluginSettingTab, Setting, TFile, TAbstractFile } from 'obsidian';

interface BDHSyncSettings {
  serverUrl: string;
  debounceMs: number;
  enabled: boolean;
  logEnabled: boolean;
}

const DEFAULT_SETTINGS: BDHSyncSettings = {
  serverUrl: 'http://localhost:8643',
  debounceMs: 1000,
  enabled: true,
  logEnabled: true,
};

// Export for testing
export const getFetch = () => globalThis.fetch;

export default class BDHSyncPlugin extends Plugin {
  settings: BDHSyncSettings = DEFAULT_SETTINGS;
  private debounceTimers: Map<string, ReturnType<typeof setTimeout>> = new Map();
  private lastSyncTime: Map<string, number> = new Map();
  private statusBar: HTMLElement | null = null;

  async onload() {
    await this.loadSettings();

    // Add status bar
    this.statusBar = this.addStatusBarItem();
    this.updateStatusBar('idle');

    // Register vault event handlers
    this.registerEvent(
      this.app.vault.on('create', (file) => this.handleFileChange(file, 'create'))
    );

    this.registerEvent(
      this.app.vault.on('modify', (file) => this.handleFileChange(file, 'modify'))
    );

    this.registerEvent(
      this.app.vault.on('delete', (file) => this.handleFileChange(file, 'delete'))
    );

    this.registerEvent(
      this.app.vault.on('rename', (file, oldPath) => this.handleFileRename(file, oldPath))
    );

    // Add settings tab
    this.addSettingTab(new BDHSyncSettingTab(this.app, this));

    console.log('BDH Graph Harness Sync plugin loaded');
  }

  onunload() {
    // Clear all pending timers
    this.debounceTimers.forEach(timer => clearTimeout(timer));
    this.debounceTimers.clear();
    console.log('BDH Graph Harness Sync plugin unloaded');
  }

  handleFileChange(file: TAbstractFile | any, eventType: 'create' | 'modify' | 'delete') {
    if (!this.settings.enabled) return;
    
    // Duck type check for TFile-like objects
    if (!file || typeof file.path !== 'string' || typeof file.extension !== 'string') return;
    if (file.extension !== 'md') return;

    // Skip files in .obsidian directory
    if (file.path.startsWith('.obsidian/')) return;

    // Debounce: wait for rapid changes to settle
    const key = `${eventType}:${file.path}`;
    const existingTimer = this.debounceTimers.get(key);
    if (existingTimer) {
      clearTimeout(existingTimer);
    }

    this.debounceTimers.set(key, setTimeout(() => {
      this.debounceTimers.delete(key);
      this.syncToServer(eventType, file.path);
    }, this.settings.debounceMs));
  }

  handleFileRename(file: TAbstractFile | any, oldPath: string) {
    if (!this.settings.enabled) return;
    
    // Duck type check for TFile-like objects
    if (!file || typeof file.path !== 'string' || typeof file.extension !== 'string') return;
    if (file.extension !== 'md') return;

    // Treat rename as delete old + create new
    this.syncToServer('delete', oldPath);
    setTimeout(() => {
      this.syncToServer('create', file.path);
    }, 100);
  }

  async syncToServer(eventType: string, filePath: string) {
    const now = Date.now();
    const lastSync = this.lastSyncTime.get(filePath) || 0;
    
    // Skip if synced too recently (prevent duplicate events)
    if (now - lastSync < 500) {
      return;
    }

    this.lastSyncTime.set(filePath, now);
    this.updateStatusBar('syncing');

    try {
      const url = `${this.settings.serverUrl}/api/node-update`;
      const fetchFn = getFetch();
      const response = await fetchFn(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          event: eventType,
          path: filePath,
        }),
      });

      if (response.ok) {
        this.updateStatusBar('ok');
        if (this.settings.logEnabled) {
          console.log(`BDH Sync: ${eventType} ${filePath}`);
        }
      } else {
        this.updateStatusBar('error');
        console.error(`BDH Sync failed: ${response.status} ${response.statusText}`);
      }
    } catch (error) {
      this.updateStatusBar('error');
      console.error('BDH Sync error:', error);
    }
  }

  updateStatusBar(status: 'idle' | 'syncing' | 'ok' | 'error') {
    if (!this.statusBar) return;

    const icons: Record<string, string> = {
      idle: '○',
      syncing: '◎',
      ok: '●',
      error: '✗',
    };

    const colors: Record<string, string> = {
      idle: 'var(--text-muted)',
      syncing: 'var(--text-accent)',
      ok: 'var(--text-success)',
      error: 'var(--text-error)',
    };

    this.statusBar.setText(`BDH ${icons[status]}`);
    this.statusBar.style.color = colors[status];
  }

  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }

  async saveSettings() {
    await this.saveData(this.settings);
  }
}

class BDHSyncSettingTab extends PluginSettingTab {
  plugin: BDHSyncPlugin;

  constructor(app: App, plugin: BDHSyncPlugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display(): void {
    const { containerEl } = this;
    containerEl.empty();

    containerEl.createEl('h2', { text: 'BDH Graph Harness Sync Settings' });

    new Setting(containerEl)
      .setName('Server URL')
      .setDesc('URL of the BDH Graph Harness server')
      .addText(text =>
        text
          .setPlaceholder('http://localhost:8643')
          .setValue(this.plugin.settings.serverUrl)
          .onChange(async (value) => {
            this.plugin.settings.serverUrl = value;
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName('Debounce delay (ms)')
      .setDesc('Wait time after last change before syncing (ms)')
      .addText(text =>
        text
          .setPlaceholder('1000')
          .setValue(String(this.plugin.settings.debounceMs))
          .onChange(async (value) => {
            const num = parseInt(value, 10);
            if (!isNaN(num) && num >= 0) {
              this.plugin.settings.debounceMs = num;
              await this.plugin.saveSettings();
            }
          })
      );

    new Setting(containerEl)
      .setName('Enable sync')
      .setDesc('Enable/disable automatic sync')
      .addToggle(toggle =>
        toggle
          .setValue(this.plugin.settings.enabled)
          .onChange(async (value) => {
            this.plugin.settings.enabled = value;
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName('Enable logging')
      .setDesc('Log sync events to console')
      .addToggle(toggle =>
        toggle
          .setValue(this.plugin.settings.logEnabled)
          .onChange(async (value) => {
            this.plugin.settings.logEnabled = value;
            await this.plugin.saveSettings();
          })
      );
  }
}
