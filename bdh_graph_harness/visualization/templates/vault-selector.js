// ============================================================================
// Vault selection — one visualisation session is always scoped to one vault.
// ============================================================================
let activeVaultId = null;

function getActiveVaultId() {
  return activeVaultId;
}

function setActiveVaultId(vaultId) {
  activeVaultId = vaultId || null;
}

function vaultApiUrl(path) {
  if (!activeVaultId) return path;
  const separator = path.includes('?') ? '&' : '?';
  return `${path}${separator}vault_id=${encodeURIComponent(activeVaultId)}`;
}

function isActiveVaultEvent(event) {
  // Server heartbeat/legacy events omit vault_id. Vault-scoped data must match.
  return !event || !event.vault_id || event.vault_id === activeVaultId;
}

function setVaultSelectorStatus(message, isError = false) {
  const status = document.getElementById('vault-selector-status');
  if (!status) return;
  status.textContent = message || '';
  status.classList.toggle('error', Boolean(isError));
}

async function loadVaultSelector() {
  const control = document.getElementById('vault-control');
  const selector = document.getElementById('vault-selector');
  if (!control || !selector) return;

  try {
    const response = await fetch('/api/vaults');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const vaults = Array.isArray(data.vaults) ? data.vaults : [];
    if (vaults.length === 0) throw new Error('No vaults configured');

    selector.replaceChildren();
    vaults.forEach(vault => {
      const option = document.createElement('option');
      option.value = vault.id;
      option.textContent = vault.name || vault.id;
      option.title = `${vault.neurons || 0} neurons · ${vault.path || vault.id}`;
      selector.appendChild(option);
    });

    const selected = vaults.some(vault => vault.id === data.default_vault)
      ? data.default_vault
      : vaults[0].id;
    setActiveVaultId(selected);
    selector.value = selected;
    control.hidden = vaults.length < 2;
    setVaultSelectorStatus('');
  } catch (error) {
    // Old single-vault servers do not expose /api/vaults. Keep visualization live.
    control.hidden = true;
    setVaultSelectorStatus('');
    console.warn('Vault selector unavailable:', error.message);
  }
}

function clearVaultView() {
  const response = document.getElementById('response-text');
  const activated = document.getElementById('activated-list');
  const queryInput = document.getElementById('query-input');
  if (response) response.textContent = '—';
  if (activated) activated.innerHTML = '<div class="empty">No activations yet</div>';
  if (queryInput) queryInput.value = '';

  if (typeof resetQuery === 'function') resetQuery();
}

function switchVault(vaultId) {
  if (!vaultId || vaultId === activeVaultId) return;
  setActiveVaultId(vaultId);
  clearVaultView();
  setVaultSelectorStatus('Loading vault…');

  if (typeof closeActiveWebSocket === 'function') closeActiveWebSocket();
  if (typeof connectWS === 'function') connectWS();
}

window.addEventListener('DOMContentLoaded', loadVaultSelector);
