/**
 * RAG Document Bot & LLM Observability Frontend Logic
 */

// State
let currentTab = 'chat';
let isQuerying = false;
let metricsInterval = null;

// Initialize on Load
document.addEventListener('DOMContentLoaded', () => {
  initDropzone();
  checkHealth();
  loadModels();
  loadDocuments();
  loadMetrics();

  // Setup periodic health & metrics refresh
  setInterval(checkHealth, 15000);
});

// Tab Switching
function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-view').forEach(v => v.classList.remove('active'));

  if (tab === 'chat') {
    document.getElementById('navChatBtn').classList.add('active');
    document.getElementById('chatView').classList.add('active');
    if (metricsInterval) {
      clearInterval(metricsInterval);
      metricsInterval = null;
    }
  } else {
    document.getElementById('navDashboardBtn').classList.add('active');
    document.getElementById('dashboardView').classList.add('active');
    loadMetrics();
    // Auto-refresh metrics every 8 seconds while on dashboard
    if (!metricsInterval) {
      metricsInterval = setInterval(loadMetrics, 8000);
    }
  }
}

// Health Check
async function checkHealth() {
  const pill = document.getElementById('healthPill');
  const txt = document.getElementById('healthText');
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    if (data.status === 'healthy') {
      pill.className = 'status-pill';
      txt.textContent = `OmniRoute Online (${data.vector_db.total_chunks} Chunks)`;
    } else {
      pill.className = 'status-pill degraded';
      txt.textContent = 'OmniRoute Degraded';
    }
  } catch (err) {
    pill.className = 'status-pill degraded';
    txt.textContent = 'Backend Offline';
  }
}

// Models Catalog
async function loadModels() {
  const select = document.getElementById('modelSelect');
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    select.innerHTML = '';
    
    data.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = `${m.name} [${m.tier}]`;
      if (m.id === data.default) opt.selected = true;
      select.appendChild(opt);
    });

    onModelChange();
  } catch (e) {
    console.error('Failed to load models:', e);
  }
}

function onModelChange() {
  const select = document.getElementById('modelSelect');
  const label = document.getElementById('inputModelLabel');
  const selectedText = select.options[select.selectedIndex]?.text || select.value;
  label.textContent = selectedText.split(' [')[0];
}

// Documents Knowledge Base
async function loadDocuments() {
  const list = document.getElementById('documentList');
  const chunksSpan = document.getElementById('statChunksCount');
  const docsSpan = document.getElementById('statDocsCount');

  try {
    const res = await fetch('/api/documents');
    const data = await res.json();

    chunksSpan.textContent = data.total_chunks || 0;
    docsSpan.textContent = data.distinct_sources?.length || (data.files?.length || 0);

    list.innerHTML = '';
    const sources = data.distinct_sources || [];

    if (sources.length === 0 && (!data.files || data.files.length === 0)) {
      list.innerHTML = '<li class="doc-item loading">No documents uploaded yet</li>';
      return;
    }

    // Display distinct indexed sources
    sources.forEach(src => {
      const li = document.createElement('li');
      li.className = 'doc-item';
      li.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
        </svg>
        <span title="${src}">${src}</span>
      `;
      list.appendChild(li);
    });
  } catch (e) {
    console.error('Failed to load documents:', e);
    list.innerHTML = '<li class="doc-item loading">Error loading documents</li>';
  }
}

// PDF Ingestion & Drag and Drop
function initDropzone() {
  const dropzone = document.getElementById('dropzone');

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      uploadFile(files[0]);
    }
  });
}

function handleFileUpload(event) {
  const file = event.target.files[0];
  if (file) {
    uploadFile(file);
  }
}

async function uploadFile(file) {
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    showFeedback('Please select a valid PDF file.', 'error');
    return;
  }

  const formData = new FormData();
  formData.append('file', file);

  const progressBar = document.getElementById('uploadProgress');
  progressBar.style.display = 'block';
  progressBar.style.width = '30%';

  showFeedback(`Ingesting & chunking "${file.name}"...`, 'info');

  try {
    progressBar.style.width = '70%';
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });

    progressBar.style.width = '100%';
    const data = await res.json();

    if (res.ok && data.success) {
      showFeedback(`Successfully indexed ${data.chunks_count} chunks from ${data.pages} page(s)!`, 'success');
      loadDocuments();
    } else {
      showFeedback(data.detail || data.message || 'Error indexing PDF', 'error');
    }
  } catch (err) {
    showFeedback(`Upload failed: ${err.message}`, 'error');
  } finally {
    setTimeout(() => {
      progressBar.style.display = 'none';
      progressBar.style.width = '0%';
    }, 1200);
  }
}

function showFeedback(msg, type) {
  const fb = document.getElementById('uploadFeedback');
  fb.textContent = msg;
  fb.className = `upload-feedback ${type}`;
  fb.style.display = 'block';
}

// Quick Prompt Click
function askQuickPrompt(text) {
  document.getElementById('chatInput').value = text;
  submitChat();
}

function handleTextareaKeydown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    submitChat(event);
  }
}

// Chat Flow
async function submitChat(event) {
  if (event) event.preventDefault();
  if (isQuerying) return;

  const input = document.getElementById('chatInput');
  const query = input.value.trim();
  if (!query) return;

  const model = document.getElementById('modelSelect').value;

  // Append User Message to UI
  appendMessage({
    role: 'user',
    content: query
  });

  input.value = '';
  isQuerying = true;
  document.getElementById('sendBtn').disabled = true;

  // Add Typing Indicator
  const typingId = 'typing-' + Date.now();
  appendTypingIndicator(typingId);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        model: model,
        top_k: 3
      })
    });

    const data = await res.json();
    removeTypingIndicator(typingId);

    if (res.ok && data.success) {
      appendMessage({
        role: 'assistant',
        content: data.answer,
        citations: data.citations,
        telemetry: data.telemetry,
        model: data.model_resolved || data.model_requested
      });
      // Background update of metrics
      loadMetrics();
    } else {
      appendMessage({
        role: 'assistant',
        content: data.answer || data.detail || 'An error occurred while generating the answer.',
        error: true,
        telemetry: data.telemetry
      });
    }
  } catch (err) {
    removeTypingIndicator(typingId);
    appendMessage({
      role: 'assistant',
      content: `Failed to connect to the server: ${err.message}`,
      error: true
    });
  } finally {
    isQuerying = false;
    document.getElementById('sendBtn').disabled = false;
    document.getElementById('chatInput').focus();
  }
}

function appendMessage({ role, content, citations, telemetry, model, error }) {
  const container = document.getElementById('chatMessages');
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role}-message`;

  if (role === 'user') {
    msgDiv.innerHTML = `
      <div class="message-body">
        <div class="message-content"><p>${escapeHtml(content)}</p></div>
      </div>
      <div class="message-avatar">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
          <circle cx="12" cy="7" r="4"></circle>
        </svg>
      </div>
    `;
  } else {
    // Assistant message with Citations and Telemetry Badge
    let citationsHtml = '';
    if (citations && citations.length > 0) {
      citationsHtml = `
        <div class="citations-container">
          <div class="citations-header" onclick="this.nextElementSibling.classList.toggle('hidden')">
            <span>Retrieved Sources (${citations.length})</span>
            <span style="font-size:0.7rem; color:var(--accent-primary);">View Chunks ▾</span>
          </div>
          <div class="citations-grid">
            ${citations.map(c => `
              <div class="citation-card">
                <div class="citation-top">
                  <span class="citation-source">[Source ${c.index}] ${escapeHtml(c.source)} (p. ${c.page})</span>
                  <span class="citation-match">${Math.round(c.similarity * 100)}% match</span>
                </div>
                <div class="citation-snippet">"${escapeHtml(c.snippet)}..."</div>
              </div>
            `).join('')}
          </div>
        </div>
      `;
    }

    let telemetryHtml = '';
    if (telemetry) {
      telemetryHtml = `
        <div class="telemetry-pill">
          <span class="telemetry-item latency">⚡ ${telemetry.latency_ms} ms</span>
          <span class="telemetry-item tokens">🪙 ${telemetry.total_tokens || 0} tokens</span>
          <span class="telemetry-item cost">💰 $${Number(telemetry.estimated_cost_usd || 0).toFixed(5)}</span>
          <span style="color:var(--text-dim);">|</span>
          <span style="color:var(--accent-cyan); font-weight:500;">${escapeHtml(model || '')}</span>
        </div>
      `;
    }

    msgDiv.innerHTML = `
      <div class="message-avatar">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="11" width="18" height="10" rx="2"></rect>
          <circle cx="12" cy="5" r="2"></circle>
          <path d="M12 7v4"></path>
        </svg>
      </div>
      <div class="message-body">
        <div class="message-meta-header">
          <span class="agent-name">DocuRAG Grounded Response</span>
          ${error ? '<span class="status-badge fail">Error</span>' : '<span class="status-badge success">Grounded</span>'}
        </div>
        <div class="message-content">${formatMarkdown(content)}</div>
        ${citationsHtml}
        ${telemetryHtml}
      </div>
    `;
  }

  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
}

function appendTypingIndicator(id) {
  const container = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.id = id;
  div.className = 'message assistant-message';
  div.innerHTML = `
    <div class="message-avatar">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"></circle>
      </svg>
    </div>
    <div class="message-body" style="padding: 12px 18px;">
      <div style="display:flex; align-items:center; gap:8px; font-size:0.82rem; color:var(--text-muted);">
        <span class="status-dot" style="background:#6366f1; animation:pulse 1.2s infinite;"></span>
        Retrieving vector chunks & generating grounded response...
      </div>
    </div>
  `;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function removeTypingIndicator(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// Dashboard Metrics Fetch & Render
async function loadMetrics() {
  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();

    const totals = data.totals || {};
    document.getElementById('kpiTotalCalls').textContent = totals.total_requests || 0;
    document.getElementById('kpiTotalCost').textContent = `$${Number(totals.total_cost_usd || 0).toFixed(5)}`;
    document.getElementById('kpiAvgLatency').textContent = `${Math.round(totals.avg_latency_ms || 0)} ms`;
    document.getElementById('kpiTotalTokens').textContent = (totals.total_tokens || 0).toLocaleString();
    document.getElementById('kpiSuccessRate').textContent = `${totals.success_rate_percent || 100}% Success Rate`;
    document.getElementById('kpiTokenSplit').textContent = `Prompt: ${(totals.prompt_tokens || 0).toLocaleString()} | Comp: ${(totals.completion_tokens || 0).toLocaleString()}`;

    // Recommendations
    const recList = document.getElementById('recList');
    if (data.recommendations && data.recommendations.length > 0) {
      recList.innerHTML = data.recommendations.map(r => `
        <div class="rec-item">
          <strong>${escapeHtml(r.title)}</strong>
          <p>${escapeHtml(r.description)}</p>
        </div>
      `).join('');
    } else {
      recList.innerHTML = '<p class="text-muted">Accumulating request data. Optimization suggestions will appear automatically.</p>';
    }

    // Model Comparison Table
    const modelTableBody = document.getElementById('modelComparisonBody');
    if (data.model_comparison && data.model_comparison.length > 0) {
      modelTableBody.innerHTML = data.model_comparison.map(m => `
        <tr>
          <td><strong style="color:var(--text-main);">${escapeHtml(m.model)}</strong></td>
          <td class="font-mono">${m.requests}</td>
          <td class="font-mono">$${Number(m.cost_usd).toFixed(4)}</td>
          <td class="font-mono">$${Number(m.avg_cost_per_query).toFixed(5)}</td>
          <td class="font-mono">${m.avg_latency_ms} ms</td>
          <td><span class="status-badge success">${m.success_rate}%</span></td>
        </tr>
      `).join('');
    } else {
      modelTableBody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No telemetry logged yet.</td></tr>';
    }

    // Recent Logs Table
    const recentBody = document.getElementById('recentLogsBody');
    if (data.recent_logs && data.recent_logs.length > 0) {
      recentBody.innerHTML = data.recent_logs.map(log => {
        const timeFormatted = log.timestamp ? log.timestamp.split('T')[1].replace('Z', '').slice(0, 8) : '';
        return `
          <tr>
            <td class="font-mono" style="font-size:0.75rem;">${timeFormatted}</td>
            <td title="${escapeHtml(log.query)}">${escapeHtml(log.query.length > 35 ? log.query.slice(0, 35) + '...' : log.query)}</td>
            <td style="font-size:0.75rem; color:var(--accent-cyan);">${escapeHtml(log.model)}</td>
            <td class="font-mono">${log.latency_ms} ms</td>
            <td class="font-mono">${log.tokens.total}</td>
            <td class="font-mono">$${Number(log.cost_usd).toFixed(5)}</td>
            <td><span class="status-badge ${log.success ? 'success' : 'fail'}">${log.success ? 'Success' : 'Error'}</span></td>
          </tr>
        `;
      }).join('');
    } else {
      recentBody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No requests recorded yet.</td></tr>';
    }
  } catch (err) {
    console.error('Failed to load metrics:', err);
  }
}

// Helpers
function escapeHtml(text) {
  if (!text) return '';
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function formatMarkdown(text) {
  if (!text) return '';
  let html = escapeHtml(text);

  // Headers
  html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h3>$1</h3>');
  html = html.replace(/^# (.*$)/gim, '<h3>$1</h3>');

  // Bold
  html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');

  // Citations highlight: [Source X: ...]
  html = html.replace(/\[Source (\d+):? ([^\]]+)\]/gim, '<span class="agent-badge" style="display:inline-block; margin:2px 4px;">Source $1: $2</span>');

  // Bullet items
  html = html.replace(/^\- (.*$)/gim, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/gims, '<ul>$1</ul>');

  // Paragraph breaks
  html = html.replace(/\n\n/g, '</p><p>');
  return `<p>${html}</p>`;
}
