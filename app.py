import pickle
import tempfile
import os
import traceback
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exception_handlers import http_exception_handler

from preprocess import preprocess_pdf, extract_image_features, TextSelector, ImgFeatureSelector

MODEL_PATH = 'models/model.pkl'
BERT_MODEL_PATH = 'models/bert'

app = FastAPI(title="作业评分系统")
_model = None        # TF-IDF + sklearn pipeline
_bert_model = None   # BERT classifier


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return JSON for all unhandled exceptions so the browser can parse the error."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


def get_model():
    global _model
    if _model is None:
        if not Path(MODEL_PATH).exists():
            raise HTTPException(status_code=503, detail="传统模型尚未训练，请先运行 python train.py")
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
    return _model


def get_bert_model():
    global _bert_model
    if _bert_model is None:
        bert_pkl = Path(BERT_MODEL_PATH) / 'bert_classifier.pkl'
        if not bert_pkl.exists():
            raise HTTPException(status_code=503, detail="BERT 模型尚未训练，请先运行 python train_bert.py")
        with open(bert_pkl, 'rb') as f:
            _bert_model = pickle.load(f)
    return _bert_model


HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>作业评分系统</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    background: #f0f2f5;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
  }
  .card {
    background: #fff;
    border-radius: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    width: 100%;
    max-width: 580px;
    padding: 40px;
  }
  h1 { font-size: 22px; font-weight: 700; color: #1a1a2e; margin-bottom: 6px; }
  .subtitle { color: #6b7280; font-size: 14px; margin-bottom: 24px; }

  /* Mode selector */
  .mode-group { display: flex; gap: 8px; margin-bottom: 24px; flex-wrap: wrap; }
  .mode-btn {
    flex: 1;
    padding: 9px 6px;
    border: 2px solid #e5e7eb;
    border-radius: 10px;
    background: #f9fafb;
    font-size: 13px;
    font-weight: 600;
    color: #6b7280;
    cursor: pointer;
    text-align: center;
    transition: all 0.15s;
    min-width: 80px;
  }
  .mode-btn:hover { border-color: #a5b4fc; color: #4338ca; }
  .mode-btn.active { border-color: #4f46e5; background: #eef2ff; color: #4338ca; }

  .drop-zone {
    border: 2px dashed #d1d5db;
    border-radius: 12px;
    padding: 36px 24px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    background: #fafafa;
  }
  .drop-zone:hover, .drop-zone.dragover { border-color: #4f46e5; background: #eef2ff; }
  .drop-zone .icon { font-size: 36px; margin-bottom: 10px; }
  .drop-zone p { color: #6b7280; font-size: 14px; line-height: 1.6; }
  .drop-zone strong { color: #4f46e5; }
  #file-input { display: none; }
  .file-name { margin-top: 10px; font-size: 13px; color: #374151; font-weight: 500; }
  .btn {
    display: block; width: 100%; margin-top: 18px; padding: 13px;
    background: #4f46e5; color: #fff; border: none; border-radius: 10px;
    font-size: 15px; font-weight: 600; cursor: pointer; transition: background 0.2s;
  }
  .btn:hover { background: #4338ca; }
  .btn:disabled { background: #9ca3af; cursor: not-allowed; }
  .spinner {
    display: none; text-align: center; margin-top: 22px;
    color: #6b7280; font-size: 14px;
  }
  .spinner::before {
    content: ''; display: inline-block; width: 18px; height: 18px;
    border: 3px solid #e5e7eb; border-top-color: #4f46e5;
    border-radius: 50%; animation: spin 0.8s linear infinite;
    margin-right: 8px; vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .result { display: none; margin-top: 26px; padding-top: 26px; border-top: 1px solid #f3f4f6; }
  .grade-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 68px; height: 68px; border-radius: 50%;
    font-size: 30px; font-weight: 800; margin-bottom: 14px;
  }
  .grade-A { background: #dcfce7; color: #166534; }
  .grade-B { background: #dbeafe; color: #1e40af; }
  .grade-C { background: #fef9c3; color: #854d0e; }
  .grade-D { background: #fee2e2; color: #991b1b; }
  .result-header { display: flex; align-items: center; gap: 14px; margin-bottom: 18px; }
  .result-info h2 { font-size: 18px; font-weight: 700; color: #111827; }
  .result-info p { font-size: 13px; color: #6b7280; margin-top: 2px; }
  .reasoning-box {
    margin-bottom: 16px; padding: 10px 14px;
    background: #f8fafc; border-radius: 8px;
    font-size: 13px; color: #374151; line-height: 1.6;
    border-left: 3px solid #4f46e5;
  }
  .bars { display: flex; flex-direction: column; gap: 10px; }
  .bar-row { display: flex; align-items: center; gap: 10px; }
  .bar-label { width: 20px; font-weight: 700; font-size: 14px; color: #374151; text-align: center; }
  .bar-track { flex: 1; background: #f3f4f6; border-radius: 6px; height: 22px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 6px; transition: width 0.6s ease; display: flex; align-items: center; padding-left: 8px; }
  .bar-fill span { font-size: 12px; font-weight: 600; color: #fff; white-space: nowrap; }
  .bar-pct { width: 42px; text-align: right; font-size: 13px; color: #6b7280; }
  .bar-A .bar-fill { background: #22c55e; }
  .bar-B .bar-fill { background: #3b82f6; }
  .bar-C .bar-fill { background: #f59e0b; }
  .bar-D .bar-fill { background: #ef4444; }
  .source-tag {
    display: inline-block; margin-top: 4px; padding: 2px 8px;
    border-radius: 20px; font-size: 11px; font-weight: 600;
    background: #e0e7ff; color: #4338ca;
  }
  .error-msg {
    display: none; margin-top: 14px; padding: 12px 16px;
    background: #fef2f2; border: 1px solid #fecaca;
    border-radius: 8px; color: #b91c1c; font-size: 13px;
  }
</style>
</head>
<body>
<div class="card">
  <h1>作业评分系统</h1>
  <p class="subtitle">上传 PDF 格式的作业文件，自动预测评分等级（A / B / C / D）</p>

  <div class="mode-group" id="mode-group">
    <div class="mode-btn active" data-mode="tfidf" onclick="selectMode(this)">TF-IDF</div>
    <div class="mode-btn" data-mode="bert"  onclick="selectMode(this)">BERT</div>
    <div class="mode-btn" data-mode="claude" onclick="selectMode(this)">Claude LLM</div>
    <div class="mode-btn" data-mode="ensemble" onclick="selectMode(this)">BERT + Claude</div>
  </div>

  <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
    <div class="icon">📄</div>
    <p><strong>点击选择</strong>或将 PDF 文件拖放至此处</p>
    <p style="margin-top:4px">仅支持 .pdf 格式</p>
    <div class="file-name" id="file-name"></div>
  </div>
  <input type="file" id="file-input" accept=".pdf">

  <button class="btn" id="submit-btn" disabled onclick="submitFile()">开始评分</button>

  <div class="spinner" id="spinner">正在分析，请稍候...</div>
  <div class="error-msg" id="error-msg"></div>

  <div class="result" id="result">
    <div class="result-header">
      <div class="grade-badge" id="grade-badge">A</div>
      <div class="result-info">
        <h2 id="result-title">评分等级：A</h2>
        <p id="result-conf">置信度：--</p>
        <span class="source-tag" id="source-tag">TF-IDF</span>
      </div>
    </div>
    <div class="reasoning-box" id="reasoning-box" style="display:none"></div>
    <div class="bars" id="bars"></div>
  </div>
</div>

<script>
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const submitBtn = document.getElementById('submit-btn');
const fileName  = document.getElementById('file-name');
let selectedFile = null;
let currentMode  = 'tfidf';

function selectMode(el) {
  document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
  el.classList.add('active');
  currentMode = el.dataset.mode;
  hideResult(); hideError();
}

fileInput.addEventListener('change', e => { if (e.target.files[0]) selectFile(e.target.files[0]); });
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('dragover');
  if (e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0]);
});

function selectFile(file) {
  if (!file.name.toLowerCase().endsWith('.pdf')) { showError('请选择 PDF 文件'); return; }
  selectedFile = file;
  fileName.textContent = '已选择：' + file.name;
  submitBtn.disabled = false;
  hideResult();
}

async function submitFile() {
  if (!selectedFile) return;
  const form = new FormData();
  form.append('file', selectedFile);
  form.append('mode', currentMode);
  submitBtn.disabled = true;
  document.getElementById('spinner').style.display = 'block';
  hideResult(); hideError();
  try {
    const resp = await fetch('/predict', { method: 'POST', body: form });
    let data;
    try { data = await resp.json(); }
    catch { throw new Error(`服务器返回非 JSON 响应 (HTTP ${resp.status})`); }
    if (!resp.ok) throw new Error(data.detail || '预测失败');
    showResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    document.getElementById('spinner').style.display = 'none';
    submitBtn.disabled = false;
  }
}

function showResult(data) {
  const grade = data.grade;
  document.getElementById('grade-badge').textContent = grade;
  document.getElementById('grade-badge').className = 'grade-badge grade-' + grade;
  document.getElementById('result-title').textContent = '评分等级：' + grade;
  document.getElementById('result-conf').textContent = '置信度：' + (data.confidence * 100).toFixed(1) + '%';
  document.getElementById('source-tag').textContent = data.source || currentMode;

  const rbx = document.getElementById('reasoning-box');
  if (data.reasoning) { rbx.textContent = 'AI 评语：' + data.reasoning; rbx.style.display = 'block'; }
  else { rbx.style.display = 'none'; }

  const barsEl = document.getElementById('bars');
  barsEl.innerHTML = '';
  const probs = data.probabilities;
  ['A','B','C','D'].forEach(g => {
    const pct = ((probs[g] || 0) * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = 'bar-row bar-' + g;
    row.innerHTML = `
      <div class="bar-label">${g}</div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${pct}%">
          ${pct > 8 ? '<span>' + pct + '%</span>' : ''}
        </div>
      </div>
      <div class="bar-pct">${pct}%</div>`;
    barsEl.appendChild(row);
  });
  document.getElementById('result').style.display = 'block';
}

function hideResult() { document.getElementById('result').style.display = 'none'; }
function showError(msg) {
  const el = document.getElementById('error-msg');
  el.textContent = '错误：' + msg;
  el.style.display = 'block';
}
function hideError() { document.getElementById('error-msg').style.display = 'none'; }
</script>
</body>
</html>
"""

GRADES = ['A', 'B', 'C', 'D']


def _probs_to_dict(probs, classes) -> dict:
    return {c: float(p) for c, p in zip(classes, probs)}


def _ensemble_probs(bert_probs: dict, claude_probs: dict, bert_weight: float = 0.4) -> dict:
    """Weighted average of BERT and Claude probability dicts."""
    cw = 1.0 - bert_weight
    merged = {}
    for g in GRADES:
        merged[g] = bert_weight * bert_probs.get(g, 0.0) + cw * claude_probs.get(g, 0.0)
    total = sum(merged.values())
    return {g: round(merged[g] / total, 4) for g in GRADES}


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.post("/predict")
async def predict(file: UploadFile = File(...), mode: str = Form("tfidf")):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="请上传 PDF 文件")

    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        text = preprocess_pdf(tmp_path)
        img_feat = extract_image_features(tmp_path)

        if mode == 'tfidf':
            model = get_model()
            X = [(text, img_feat)]
            grade = model.predict(X)[0]
            probs = model.predict_proba(X)[0]
            probs_dict = _probs_to_dict(probs, model.classes_)
            return JSONResponse({
                'grade': grade,
                'confidence': float(max(probs)),
                'probabilities': probs_dict,
                'source': 'TF-IDF',
            })

        elif mode == 'bert':
            if not text.strip():
                raise HTTPException(status_code=422, detail="无法从 PDF 中提取文本")
            bert = get_bert_model()
            probs = bert.predict_proba([text])[0]
            classes = bert.classes_
            grade = classes[probs.argmax()]
            probs_dict = _probs_to_dict(probs, classes)
            return JSONResponse({
                'grade': grade,
                'confidence': float(max(probs)),
                'probabilities': probs_dict,
                'source': 'BERT',
            })

        elif mode == 'claude':
            from claude_processor import grade_pdf_with_claude
            result = grade_pdf_with_claude(tmp_path, use_vision=True)
            return JSONResponse({
                'grade': result['grade'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'reasoning': result.get('reasoning', ''),
                'source': 'Claude',
            })

        elif mode == 'ensemble':
            if not text.strip():
                raise HTTPException(status_code=422, detail="无法从 PDF 中提取文本")

            # BERT on plain text (no Claude token cost)
            from claude_processor import grade_pdf_with_claude
            bert = get_bert_model()
            bert_probs_arr = bert.predict_proba([text])[0]
            bert_probs = _probs_to_dict(bert_probs_arr, bert.classes_)

            # Claude grades the PDF (vision included) and provides reasoning
            claude_result = grade_pdf_with_claude(tmp_path, use_vision=True)
            claude_probs = claude_result['probabilities']

            # Weighted average (BERT 40%, Claude 60%)
            merged = _ensemble_probs(bert_probs, claude_probs, bert_weight=0.4)
            best_grade = max(merged, key=merged.get)
            return JSONResponse({
                'grade': best_grade,
                'confidence': float(merged[best_grade]),
                'probabilities': merged,
                'reasoning': claude_result.get('reasoning', ''),
                'source': 'BERT + Claude',
            })

        else:
            raise HTTPException(status_code=400, detail=f"未知模式: {mode}")

    finally:
        os.unlink(tmp_path)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
