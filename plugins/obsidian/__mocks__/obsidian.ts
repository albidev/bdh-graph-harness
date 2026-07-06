// Mock for obsidian module
export class App {
  vault = {
    on: jest.fn(),
    getAbstractFileByPath: jest.fn(),
  };
}

export class Plugin {
  app: App = new App();
  loadData = jest.fn().mockResolvedValue({});
  saveData = jest.fn().mockResolvedValue(undefined);
  addStatusBarItem = jest.fn().mockReturnValue({
    setText: jest.fn(),
    style: { color: '' },
  });
  registerEvent = jest.fn();
  addSettingTab = jest.fn();
}

export class PluginSettingTab {
  app: App;
  plugin: Plugin;

  constructor(app: App, plugin: Plugin) {
    this.app = app;
    this.plugin = plugin;
  }
}

export class Setting {
  constructor(containerEl: HTMLElement) {}

  setName(name: string) {
    return this;
  }

  setDesc(desc: string) {
    return this;
  }

  addText(callback: (text: any) => void) {
    return this;
  }

  addToggle(callback: (toggle: any) => void) {
    return this;
  }
}

export class TFile {
  path: string = '';
  extension: string = '';
}

export class TAbstractFile {
  path: string = '';
}

// Mock fetch globally
global.fetch = jest.fn().mockResolvedValue({
  ok: true,
  status: 200,
  statusText: 'OK',
} as Response);
