#!/usr/bin/env python3
"""
====================================================================
  VEDAX COMPLETE SERVER — Knowledge Management Agent
  
  Features:
  - Multi-document SOP management (categorization + versioning)
  - Vectorless RAG (BM25 + HD expansion + citation verification)
  - Audit logging (every Q&A tracked for compliance)
  - Role-based access (HR/Finance/IT/etc + Admin)
  - Unanswered questions tracker (gaps in knowledge base)
  - Trending questions dashboard (what employees ask most)
  - Compliance report generator (stats + grounding %)
  - Admin dashboard (logs, stats, unanswered questions)

SETUP:
  pip install fastapi uvicorn python-multipart

RUN:
  python vedax_complete_server.py
  → http://localhost:8000  (main UI)
  → http://localhost:8000/admin (admin dashboard)
  → http://localhost:8000/docs (Swagger API docs)

DEFAULT ROLES (registered in DB):
  - admin (all categories)
  - hr-user (HR only)
  - finance-user (Finance only)
  - general-user (General only)
  
  API key format: use as header "X-API-Key: {api_key}"
  (defaults: admin-key, hr-key, finance-key, general-key)
====================================================================
"""

import os
import shutil
import secrets
import json
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import vedax_core as core
from vedax_db import db

# ────────────────────────────────────────────────────────────────
#  Setup default roles if not already registered
# ────────────────────────────────────────────────────────────────

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DEFAULT_ROLES = [
    ("admin", "admin", "Administrator", ["HR", "Finance", "IT", "Compliance", "Admin", "General"]),
    ("hr-user", "hr", "HR Employee", ["HR"]),
    ("finance-user", "finance", "Finance Employee", ["Finance"]),
    ("general-user", "general", "General User", ["General"]),
]

for user_id, role, name, cats in DEFAULT_ROLES:
    if not db.get_role(user_id):
        api_key = f"{role}-key-{secrets.token_hex(8)}"
        db.register_role(user_id, role, name, cats, api_key)

# ────────────────────────────────────────────────────────────────
#  Role-based access control
# ────────────────────────────────────────────────────────────────

def verify_api_key(x_api_key: str = Header(None)) -> dict:
    """Verify API key and return user role info."""
    if not x_api_key:
        # Demo mode: default to admin if no key provided (for local testing)
        return {
            "user_id": "demo-user",
            "role": "admin",
            "allowed_categories": ["HR", "Finance", "IT", "Compliance", "Admin", "General"],
        }
    role_info = db.get_role_by_key(x_api_key)
    if not role_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return {
        "user_id": role_info["user_id"],
        "role": role_info["role"],
        "allowed_categories": json.loads(role_info.get("allowed_categories", "[]")),
    }


def filter_categories(user_info: dict, requested_category: Optional[str]) -> Optional[str]:
    """Ensure user can only access their allowed categories."""
    if not requested_category:
        return None
    if requested_category not in user_info["allowed_categories"]:
        raise HTTPException(status_code=403, detail=f"Access denied to '{requested_category}' category")
    return requested_category


# ────────────────────────────────────────────────────────────────
#  FastAPI app
# ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VedaX KM Agent",
    description="Knowledge Management Agent for SOP documents — vectorless RAG + audit logging + admin dashboard",
    version="2.0.0",
)


# ────────────────────────────────────────────────────────────────
#  Request/Response schemas
# ────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str
    category: Optional[str] = None


class RetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    category: Optional[str] = None


class DocumentRequest(BaseModel):
    path: str
    category: Optional[str] = "General"
    tags: Optional[List[str]] = None


# ────────────────────────────────────────────────────────────────
#  Main UI (document management + Q&A)
# ────────────────────────────────────────────────────────────────

MAIN_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>VedaX KM Agent</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  :root {
    --bg: #faf9f6;
    --surface: #ffffff;
    --border: #e4e1d8;
    --text: #21241f;
    --text-soft: #6b6f64;
    --accent: #4a5d4e;
    --accent-hover: #3b4a3e;
    --danger: #a23b2e;
    --info: #1e5a96;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 48px 24px;
  }
  .wrap { max-width: 900px; margin: 0 auto; }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 32px;
    border-bottom: 2px solid var(--border);
    padding-bottom: 16px;
  }
  h1 { font-size: 1.8rem; font-weight: 700; letter-spacing: -0.01em; margin: 0; }
  .header-right { display: flex; gap: 12px; align-items: center; }
  a.btn-admin {
    padding: 8px 14px;
    background: var(--info);
    color: white;
    text-decoration: none;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 600;
  }
  a.btn-admin:hover { background: #153d6f; }
  .sub { color: var(--text-soft); font-size: 0.9rem; margin: 0 0 32px; }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
  }
  .card h2 {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-soft);
    margin: 0 0 14px;
  }
  .upload-meta-row { display: flex; gap: 10px; margin-bottom: 10px; }
  .upload-meta-row input[type="text"] { flex: 1; }
  input[type="file"] { font-size: 0.85rem; color: var(--text-soft); margin-bottom: 10px; }
  input[type="text"], select {
    font-size: 0.9rem;
    padding: 8px 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    background: var(--bg);
  }
  button {
    border: none;
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
  }
  .btn-primary { background: var(--accent); color: white; }
  .btn-primary:hover { background: var(--accent-hover); }
  .btn-primary:disabled { background: #b9c2ba; cursor: not-allowed; }
  .btn-danger { background: transparent; color: var(--danger); border: 1px solid #e4cfc9; padding: 6px 12px; }
  .btn-danger:hover { background: #fbeeec; }
  ul.doc-list { list-style: none; margin: 0; padding: 0; }
  ul.doc-list li {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid var(--border);
  }
  ul.doc-list li:last-child { border-bottom: none; }
  .doc-name { font-size: 0.9rem; font-weight: 500; }
  .doc-path { font-size: 0.78rem; color: var(--text-soft); }
  .chip-row { margin-top: 4px; display: flex; gap: 6px; flex-wrap: wrap; }
  .chip {
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 999px;
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text-soft);
  }
  .chip-category { background: #eef1ea; color: var(--accent); border-color: #d7e0d4; font-weight: 600; }
  .filter-row { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
  .filter-row label { font-size: 0.82rem; color: var(--text-soft); }
  .answer-text { font-size: 0.94rem; line-height: 1.6; }
  .answer-text p { margin: 0 0 10px; }
  .answer-text ul, .answer-text ol { margin: 0 0 10px; padding-left: 22px; }
  .answer-text li { margin-bottom: 5px; }
  .answer-text strong { font-weight: 600; }
  .badge {
    display: inline-block;
    margin-top: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 4px 9px;
    border-radius: 5px;
  }
  .badge-ok { background: #e7efe8; color: var(--accent); }
  .badge-warn { background: #fbf3df; color: #92660c; }
  .badge-bad { background: #fbeeec; color: var(--danger); }
  .sources { margin-top: 10px; font-size: 0.8rem; color: var(--text-soft); }
  .status { font-size: 0.82rem; margin-top: 12px; min-height: 1.2em; }
  .status.ok { color: var(--accent); }
  .status.err { color: var(--danger); }
  .empty { color: var(--text-soft); font-size: 0.88rem; padding: 8px 0; }
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>VedaX KM Agent</h1>
      <div class="header-right">
        <a href="/admin" class="btn-admin">📊 Admin Dashboard</a>
      </div>
    </header>

    <div class="card">
      <h2>Upload SOP Document</h2>
      <input type="file" id="fileInput" accept=".pdf,.txt" />
      <div class="upload-meta-row">
        <input type="text" id="categoryInput" list="categoryOptions" placeholder="Category (e.g. HR)" />
        <input type="text" id="tagsInput" placeholder="Tags, comma separated" />
      </div>
      <datalist id="categoryOptions"></datalist>
      <button class="btn-primary" id="uploadBtn">Upload</button>
      <div class="status" id="uploadStatus"></div>
    </div>

    <div class="card">
      <h2>Indexed Documents</h2>
      <div class="filter-row">
        <label for="filterCategory">Filter:</label>
        <select id="filterCategory"><option value="">All</option></select>
      </div>
      <ul class="doc-list" id="docList"></ul>
      <div class="status" id="listStatus"></div>
    </div>

    <div class="card">
      <h2>Ask a Question</h2>
      <div style="display: flex; gap: 10px; margin-bottom: 10px;">
        <input type="text" id="queryInput" placeholder="Apna sawaal likho..." style="flex: 1;" />
        <select id="askCategory"><option value="">All categories</option></select>
        <button class="btn-primary" id="askBtn">Ask</button>
      </div>
      <div id="answerBox"></div>
    </div>
  </div>

<script>
let allDocuments = [];

async function loadCategories() {
  try {
    const res = await fetch('/categories');
    const data = await res.json();
    const cats = data.categories || [];
    document.getElementById('categoryOptions').innerHTML = cats.map(c => '<option value="' + c + '"></option>').join('');
    [document.getElementById('filterCategory'), document.getElementById('askCategory')].forEach(sel => {
      const curr = sel.value;
      sel.innerHTML = '<option value="">All</option>' + cats.map(c => '<option value="' + c + '">' + c + '</option>').join('');
      sel.value = curr;
    });
  } catch (e) {}
}

function renderDocList() {
  const list = document.getElementById('docList');
  const filter = document.getElementById('filterCategory').value;
  const docs = filter ? allDocuments.filter(d => d.category === filter) : allDocuments;
  list.innerHTML = '';
  if (docs.length === 0) {
    list.innerHTML = '<li class="empty">Koi document nahi — upload karo.</li>';
    return;
  }
  docs.forEach(doc => {
    const li = document.createElement('li');
    const name = doc.path.split(/[\\\\/]/).pop();
    const tagsHtml = (doc.tags || []).map(t => '<span class="chip">' + t + '</span>').join('');
    const deprec = doc.deprecated ? ' (Deprecated)' : '';
    const left = document.createElement('div');
    left.innerHTML =
      '<div class="doc-name">' + name + deprec + '</div>' +
      '<div class="doc-path">' + doc.path + '</div>' +
      '<div class="chip-row"><span class="chip chip-category">' + doc.category + ' v' + (doc.version || '1.0') + '</span>' + tagsHtml + '</div>';
    const delBtn = document.createElement('button');
    delBtn.className = 'btn-danger';
    delBtn.textContent = 'Delete';
    delBtn.addEventListener('click', () => deleteDocument(doc.path));
    li.appendChild(left);
    li.appendChild(delBtn);
    list.appendChild(li);
  });
}

async function loadDocuments() {
  try {
    const res = await fetch('/documents');
    const data = await res.json();
    allDocuments = data.documents || [];
    renderDocList();
  } catch (e) {
    document.getElementById('listStatus').textContent = 'Load failed';
    document.getElementById('listStatus').className = 'status err';
  }
}

async function deleteDocument(path) {
  if (!confirm('Delete "' + path + '"?')) return;
  const status = document.getElementById('listStatus');
  status.textContent = 'Deleting...';
  try {
    const res = await fetch('/documents', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: path })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Delete failed');
    status.textContent = 'Deleted.';
    status.className = 'status ok';
    loadDocuments();
  } catch (e) {
    status.textContent = e.message;
    status.className = 'status err';
  }
}

document.getElementById('uploadBtn').addEventListener('click', async () => {
  const input = document.getElementById('fileInput');
  const cat = document.getElementById('categoryInput');
  const tags = document.getElementById('tagsInput');
  const status = document.getElementById('uploadStatus');
  const btn = document.getElementById('uploadBtn');
  if (!input.files.length) {
    status.textContent = 'Select a file first.';
    status.className = 'status err';
    return;
  }
  const fd = new FormData();
  fd.append('file', input.files[0]);
  fd.append('category', cat.value.trim() || 'General');
  fd.append('tags', tags.value.trim());
  btn.disabled = true;
  status.textContent = 'Uploading...';
  status.className = 'status';
  try {
    const res = await fetch('/documents/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Upload failed');
    status.textContent = 'Uploaded.';
    status.className = 'status ok';
    input.value = '';
    cat.value = '';
    tags.value = '';
    loadDocuments();
    loadCategories();
  } catch (e) {
    status.textContent = e.message;
    status.className = 'status err';
  } finally {
    btn.disabled = false;
  }
});

document.getElementById('filterCategory').addEventListener('change', renderDocList);

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function renderAnswer(data) {
  const box = document.getElementById('answerBox');
  if (data.error) {
    box.innerHTML = '<div class="status err">' + escapeHtml(data.error) + '</div>';
    return;
  }
  const html = typeof marked !== 'undefined'
    ? marked.parse(data.answer || '')
    : '<p>' + escapeHtml(data.answer || '') + '</p>';
  let out = '<div class="answer-text">' + html + '</div>';
  if (data.grounding) {
    const b = data.grounding.badge;
    const cls = b === 'OK' ? 'badge-ok' : b === 'WARN' ? 'badge-warn' : 'badge-bad';
    out += '<div class="badge ' + cls + '">Grounded ' + Math.round(data.grounding.grounded_fraction * 100) + '%</div>';
  }
  if (data.sources && data.sources.length) {
    out += '<div class="sources">Sources: ' + data.sources.map(s => escapeHtml(s.file)).join(', ') + '</div>';
  }
  box.innerHTML = out;
}

async function askQuestion() {
  const input = document.getElementById('queryInput');
  const cat = document.getElementById('askCategory');
  const box = document.getElementById('answerBox');
  const btn = document.getElementById('askBtn');
  const query = input.value.trim();
  if (!query) return;
  btn.disabled = true;
  box.innerHTML = '<div class="status">Soch raha hai...</div>';
  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, category: cat.value || null })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Ask failed');
    renderAnswer(data);
  } catch (e) {
    box.innerHTML = '<div class="status err">' + escapeHtml(e.message) + '</div>';
  } finally {
    btn.disabled = false;
  }
}

document.getElementById('askBtn').addEventListener('click', askQuestion);
document.getElementById('queryInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') askQuestion();
});

loadCategories();
loadDocuments();
</script>
</body>
</html>
"""


# ────────────────────────────────────────────────────────────────
#  Admin Dashboard UI
# ────────────────────────────────────────────────────────────────

ADMIN_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>VedaX Admin Dashboard</title>
<style>
  :root {
    --bg: #faf9f6;
    --surface: #ffffff;
    --border: #e4e1d8;
    --text: #21241f;
    --text-soft: #6b6f64;
    --accent: #4a5d4e;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
  }
  .wrap { max-width: 1200px; margin: 0 auto; }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
    border-bottom: 2px solid var(--border);
    padding-bottom: 12px;
  }
  h1 { font-size: 1.6rem; margin: 0; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 24px; }
  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
  }
  .stat-card h3 { margin: 0 0 12px; font-size: 0.9rem; color: var(--text-soft); text-transform: uppercase; }
  .stat-value { font-size: 2rem; font-weight: 700; color: var(--accent); }
  .stat-sub { font-size: 0.85rem; color: var(--text-soft); margin-top: 8px; }
  table { width: 100%; border-collapse: collapse; }
  th {
    text-align: left;
    padding: 12px;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    border-bottom: 2px solid var(--border);
  }
  td { padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 0.9rem; }
  tr:hover { background: var(--bg); }
  .section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
  }
  .section h2 { margin: 0 0 14px; font-size: 1.1rem; }
  .badge {
    display: inline-block;
    font-size: 0.7rem;
    padding: 3px 8px;
    border-radius: 999px;
    background: var(--bg);
    color: var(--text-soft);
  }
  .badge-ok { background: #e7efe8; color: var(--accent); }
  .badge-warn { background: #fbf3df; color: #92660c; }
  .badge-danger { background: #fbeeec; color: #a23b2e; }
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>📊 Admin Dashboard</h1>
      <a href="/">← Back to Main UI</a>
    </header>

    <div class="cards" id="statCards"></div>

    <div class="section">
      <h2>Trending Questions (Last 30 Days)</h2>
      <table id="trendingTable"><tr><th>Query</th><th>Count</th><th>Category</th></tr></table>
    </div>

    <div class="section">
      <h2>Unanswered Questions (Open)</h2>
      <table id="unansweredTable"><tr><th>Query</th><th>User</th><th>Category</th><th>Confidence</th><th>When</th></tr></table>
    </div>

    <div class="section">
      <h2>Recent Queries (Last 50)</h2>
      <table id="auditTable"><tr><th>When</th><th>User</th><th>Query</th><th>Category</th><th>Grounded</th><th>Abstained</th></tr></table>
    </div>
  </div>

<script>
async function loadDashboard() {
  // Stats
  try {
    const res = await fetch('/admin/compliance-report');
    const data = await res.json();
    const cards = document.getElementById('statCards');
    cards.innerHTML = `
      <div class="stat-card">
        <h3>Total Queries</h3>
        <div class="stat-value">${data.total_queries}</div>
        <div class="stat-sub">${data.abstained_count} abstained (${data.abstained_pct}%)</div>
      </div>
      <div class="stat-card">
        <h3>Avg Grounding</h3>
        <div class="stat-value">${data.avg_grounded_pct}%</div>
        <div class="stat-sub">of answered questions backed by chunks</div>
      </div>
      <div class="stat-card">
        <h3>Open Unanswered</h3>
        <div class="stat-value">${data.open_unanswered}</div>
        <div class="stat-sub">waiting for manual review</div>
      </div>
    `;
  } catch (e) {
    console.error(e);
  }

  // Trending
  try {
    const res = await fetch('/admin/trending-questions');
    const data = await res.json();
    const table = document.getElementById('trendingTable');
    data.questions.forEach(q => {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td>' + q.query + '</td><td>' + q.count + '</td><td>' + q.category + '</td>';
      table.appendChild(tr);
    });
  } catch (e) {
    console.error(e);
  }

  // Unanswered
  try {
    const res = await fetch('/admin/unanswered-questions');
    const data = await res.json();
    const table = document.getElementById('unansweredTable');
    data.questions.forEach(q => {
      const tr = document.createElement('tr');
      const when = new Date(q.timestamp * 1000).toLocaleString();
      tr.innerHTML = '<td>' + q.query + '</td><td>' + q.user_id + '</td><td>' + q.category + '</td><td>' + q.confidence.toFixed(2) + '</td><td>' + when + '</td>';
      table.appendChild(tr);
    });
  } catch (e) {
    console.error(e);
  }

  // Audit logs
  try {
    const res = await fetch('/admin/audit-logs?limit=50');
    const data = await res.json();
    const table = document.getElementById('auditTable');
    data.logs.forEach(log => {
      const tr = document.createElement('tr');
      const when = new Date(log.timestamp * 1000).toLocaleString();
      const grounded = log.grounded_fraction ? Math.round(log.grounded_fraction * 100) + '%' : 'N/A';
      const badge = log.abstained ? '<span class="badge badge-danger">Abstained</span>' : grounded;
      tr.innerHTML = '<td>' + when + '</td><td>' + log.user_id + '</td><td>' + log.query.substring(0, 40) + '...</td><td>' + log.category + '</td><td>' + badge + '</td><td>' + (log.abstained ? 'Yes' : 'No') + '</td>';
      table.appendChild(tr);
    });
  } catch (e) {
    console.error(e);
  }
}

loadDashboard();
</script>
</body>
</html>
"""


# ────────────────────────────────────────────────────────────────
#  Routes
# ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def main_ui():
    """Main UI — document management + Q&A."""
    return MAIN_UI_HTML


@app.get("/admin", response_class=HTMLResponse)
def admin_ui():
    """Admin dashboard — audit logs, trending questions, compliance reports."""
    return ADMIN_UI_HTML


@app.get("/status")
def status(user_info: dict = Depends(verify_api_key)):
    """System status."""
    return core.store.status()


@app.get("/documents")
def list_documents(user_info: dict = Depends(verify_api_key)):
    """List documents user can access."""
    core.store.ensure_loaded()
    all_docs = core.store.list_documents()
    user_cats = user_info["allowed_categories"]
    return {"documents": [d for d in all_docs if d["category"] in user_cats]}


@app.get("/categories")
def categories(user_info: dict = Depends(verify_api_key)):
    """Categories user can access."""
    core.store.ensure_loaded()
    all_cats = core.store.all_categories()
    user_cats = user_info["allowed_categories"]
    return {"categories": [c for c in all_cats if c in user_cats]}


@app.post("/documents")
def add_document(req: DocumentRequest, user_info: dict = Depends(verify_api_key)):
    """Add a document (server-side path)."""
    cat = filter_categories(user_info, req.category)
    try:
        seconds = core.store.add_document(req.path, category=cat or "General", tags=req.tags or [])
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "documents": core.store.list_documents(),
        "chunks_indexed": len(core.store.engine.chunks),
        "seconds": round(seconds, 2),
    }


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form("General"),
    tags: str = Form(""),
    user_info: dict = Depends(verify_api_key),
):
    """Upload a document."""
    cat = filter_categories(user_info, category)
    dest_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    try:
        seconds = core.store.add_document(dest_path, category=cat or "General", tags=tag_list)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "documents": core.store.list_documents(),
        "chunks_indexed": len(core.store.engine.chunks),
        "seconds": round(seconds, 2),
        "saved_to": dest_path,
    }


@app.delete("/documents")
def remove_document(req: DocumentRequest, user_info: dict = Depends(verify_api_key)):
    """Delete a document."""
    try:
        seconds = core.store.remove_document(req.path)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "documents": core.store.list_documents(),
        "chunks_indexed": len(core.store.engine.chunks),
        "seconds": round(seconds, 2),
    }


@app.post("/retrieve")
def retrieve(req: RetrieveRequest, user_info: dict = Depends(verify_api_key)):
    """Debug retrieval only."""
    top_k = req.top_k or core.TOP_K
    cat = filter_categories(user_info, req.category) if req.category else None
    return core.do_retrieve(req.query, top_k, category=cat)


@app.post("/ask")
def ask(req: AskRequest, user_info: dict = Depends(verify_api_key)):
    """Ask a question (logs to audit trail)."""
    cat = filter_categories(user_info, req.category) if req.category else None
    result = core.do_ask(req.query, category=cat)
    # Log to audit trail
    core.log_answer(user_info["user_id"], user_info["role"], req.query, cat, result)
    return result


# ────────────────────────────────────────────────────────────────
#  Admin endpoints
# ────────────────────────────────────────────────────────────────

@app.get("/admin/audit-logs")
def audit_logs(limit: int = 50, user_info: dict = Depends(verify_api_key)):
    """Get audit logs (admin only)."""
    if user_info["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    logs = db.get_audit_logs(days=90)[:limit]
    return {"logs": logs}


@app.get("/admin/unanswered-questions")
def unanswered_questions(user_info: dict = Depends(verify_api_key)):
    """Get unanswered questions (admin only)."""
    if user_info["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    questions = db.get_unanswered(status="open")
    return {"questions": questions}


@app.get("/admin/trending-questions")
def trending_questions(user_info: dict = Depends(verify_api_key)):
    """Get trending questions (admin only)."""
    if user_info["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    questions = db.get_trending_questions(days=30, limit=20)
    return {"questions": questions}


@app.get("/admin/compliance-report")
def compliance_report(days: int = 90, user_info: dict = Depends(verify_api_key)):
    """Get compliance report (admin only)."""
    if user_info["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return db.get_compliance_report(days=days)


# ────────────────────────────────────────────────────────────────
#  Run
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print("  VEDAX COMPLETE — Knowledge Management Agent")
    print("=" * 70)
    print("\n  🌐 Main UI:     http://localhost:8000")
    print("  📊 Admin:       http://localhost:8000/admin")
    print("  📖 API Docs:    http://localhost:8000/docs")
    print("\n  Default roles registered:")
    print("    - admin (all categories)")
    print("    - hr-user (HR only)")
    print("    - finance-user (Finance only)")
    print("    - general-user (General only)")
    print("\n  API key header: X-API-Key: {api-key}")
    print("  Demo mode: no key needed (defaults to admin)\n")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
