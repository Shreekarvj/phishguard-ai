"""
Real-Time Phishing & Scam Message Detector — Web Application
CAT-03 Case Study: Using an AI-Enabled Cloud Tool (Google Gemini API)
Author: Shreekar V J | USN: ENG24CS0220 | Section: 5D

Run locally:
    pip install flask requests
    $env:GEMINI_API_KEY="your_api_key_here"
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import json
import os
import re
from datetime import datetime
from urllib.parse import urlparse

import requests
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# Google Gemini Developer API (Google AI Studio) — tries active free-tier Flash models in order
GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
]
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
SAMPLES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_messages.json")
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_results.json")

URGENCY_PATTERNS = [
    r"\burgent\b",
    r"\bimmediately\b",
    r"\bwithin 24 hours\b",
    r"\baccount (?:suspended|locked|blocked|compromised)\b",
    r"\bverify (?:now|your account|immediately)\b",
    r"\baction required\b",
    r"\bfinal warning\b",
    r"\bexpire[sd]?\b",
]

CREDENTIAL_PATTERNS = [
    r"\botp\b",
    r"\bpassword\b",
    r"\bpin\b",
    r"\bcvv\b",
    r"\bkyc\b",
    r"\bbank details\b",
    r"\bcredit card\b",
    r"\blogin\b",
]

LURE_PATTERNS = [
    r"\bwon\b",
    r"\bprize\b",
    r"\blottery\b",
    r"\breward\b",
    r"\bfree\b",
    r"\bcrypto\b",
    r"\bguaranteed return\b",
]

SHORTENER_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "rb.gy",
    "ow.ly",
    "is.gd",
}

DEFAULT_SAMPLES = [
    {
        "id": 1,
        "category": "Bank KYC Phishing SMS",
        "message": "URGENT: Your SBI bank account has been suspended due to incomplete KYC. Verify your OTP and PAN details immediately within 24 hours at http://sbi-kyc-verify.example.com/login to avoid permanent block.",
    },
    {
        "id": 2,
        "category": "Delivery Scam SMS",
        "message": "Your parcel delivery failed due to an incomplete address. Update your details now to reschedule: https://bit.ly/3xFakeParcel",
    },
    {
        "id": 3,
        "category": "Prize / Crypto Scam",
        "message": "Congratulations! You won a 50,000 INR reward in our crypto lucky draw. Share your bank details and CVV to claim your prize immediately!",
    },
    {
        "id": 4,
        "category": "Legitimate University Notice",
        "message": "Reminder from the CSE Department: CAT-03 case study demonstrations and report submissions are scheduled for Monday, 23 November 2026 between 10:00 AM and 3:00 PM.",
    },
    {
        "id": 5,
        "category": "Legitimate Bank Alert",
        "message": "Rs. 450.00 debited from A/C XX1234 via UPI on 19-11-26. If you did not authorize this transaction, call 1800-11-2211.",
    },
]

SCAN_HISTORY: list[dict] = []


def load_samples() -> list[dict]:
    if os.path.exists(SAMPLES_PATH):
        try:
            with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return DEFAULT_SAMPLES


def extract_red_flags(text: str) -> list[str]:
    """Scan message text locally for common phishing heuristics."""
    flags = []
    lowered = text.lower()

    if any(re.search(pat, lowered) for pat in URGENCY_PATTERNS):
        flags.append("Urgency or account-suspension threat detected")

    if any(re.search(pat, lowered) for pat in CREDENTIAL_PATTERNS):
        flags.append("Requests sensitive credentials, OTP, or KYC/banking details")

    if any(re.search(pat, lowered) for pat in LURE_PATTERNS):
        flags.append("Contains prize, reward, or financial lure keywords")

    urls = re.findall(r"https?://[^\s<>\"']+", text)
    for url in urls:
        parsed = urlparse(url)
        host = parsed.netloc.lower().split(":")[0]
        if host in SHORTENER_DOMAINS:
            flags.append(f"Uses URL shortener ({host}) that hides the true destination")
        elif re.match(r"^\d{1,3}(?:\.\d{1,3}){3}$", host):
            flags.append(f"Links to a raw IP address ({host}) instead of a domain")
        elif "-" in host and any(
            brand in host for brand in ("bank", "sbi", "hdfc", "paypal", "amazon", "verify", "secure")
        ):
            flags.append(f"Suspicious hyphenated lookalike domain ({host})")
        elif parsed.scheme == "http":
            flags.append(f"Uses unencrypted HTTP link ({host})")

    return flags


def classify_with_gemini(text: str, api_key: str) -> dict:
    """Send message to Google Gemini Cloud API and parse structured JSON verdict."""
    prompt = (
        "You are a cybersecurity phishing detector. Analyze the following SMS, email, or URL message "
        "and respond ONLY with a valid JSON object using these exact keys:\n"
        '{"phishing_probability": <number 0 to 100>, '
        '"explanation": "<one concise sentence explaining why>", '
        '"red_flags": ["<short red flag 1>", "<short red flag 2>"]}\n'
        "If the message is legitimate and safe, set phishing_probability below 20 and red_flags to [].\n\n"
        f"Message:\n{text}"
    )

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
 payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1,
            "maxOutputTokens": 256,
            "thinkingConfig": {"thinkingLevel": "LOW"},
        },
    }

    last_error = "Unknown error"
    for model_name in GEMINI_MODELS:
        url = f"{GEMINI_BASE_URL}/{model_name}:generateContent"
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=12)
        except requests.Timeout:
            last_error = f"{model_name} timed out"
            continue
        if response.status_code in (400, 404, 429, 503):
            last_error = f"HTTP {response.status_code}"
            continue
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code}")

        body = response.json()
        raw_text = (
            body.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "{}")
            .strip()
        )
        # Strip markdown fences if present
        raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.S).strip()
        parsed = json.loads(raw_text)

        prob = max(0.0, min(100.0, float(parsed.get("phishing_probability", 50.0))))
        explanation = str(parsed.get("explanation", "")).strip()
        ai_flags = [str(f).strip() for f in parsed.get("red_flags", []) if str(f).strip()]

        return {
            "source": "Google Gemini API (Google AI Studio)",
            "model": model_name,
            "phishing_probability": round(prob, 2),
            "explanation": explanation,
            "ai_flags": ai_flags,
        }

    raise RuntimeError(last_error)


def local_fallback_score(flags: list[str], reason: str = "GEMINI_API_KEY not set") -> dict:
    """Fallback heuristic score used only when API key is absent or unreachable."""
    score = min(15.0 + len(flags) * 25.0, 95.0) if flags else 8.0
    return {
        "source": f"Local Heuristic Fallback ({reason})",
        "model": "Rule-Based Scanner",
        "phishing_probability": round(score, 2),
        "explanation": "Evaluated using local regex heuristics because the cloud API was not configured or unreachable.",
        "ai_flags": [],
    }


def determine_verdict(phishing_prob: float, flags: list[str]) -> str:
    if phishing_prob >= 75.0 or (phishing_prob >= 55.0 and len(flags) >= 2):
        return "HIGH-RISK PHISHING"
    if phishing_prob >= 40.0 or len(flags) >= 1:
        return "SUSPICIOUS"
    return "SAFE / LEGITIMATE"


def analyze_message(text: str, custom_key: str | None = None) -> dict:
    local_flags = extract_red_flags(text)
    api_key = (custom_key or "").strip() or os.environ.get("GEMINI_API_KEY", "").strip()

    if api_key:
        try:
            ai_result = classify_with_gemini(text, api_key)
        except Exception as exc:
            ai_result = local_fallback_score(local_flags, reason=f"Cloud error: {exc}")
    else:
        ai_result = local_fallback_score(local_flags)

    # Merge AI-detected red flags with local heuristic flags (deduplicated)
    combined_flags = list(dict.fromkeys(ai_result.get("ai_flags", []) + local_flags))
    verdict = determine_verdict(ai_result["phishing_probability"], combined_flags)

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": text,
        "message_preview": text[:100] + ("..." if len(text) > 100 else ""),
        "verdict": verdict,
        "phishing_probability_pct": ai_result["phishing_probability"],
        "inference_source": ai_result["source"],
        "model": ai_result["model"],
        "explanation": ai_result.get("explanation", ""),
        "red_flags": combined_flags,
    }
    SCAN_HISTORY.insert(0, record)
    try:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(SCAN_HISTORY, f, indent=2)
    except OSError:
        pass
    return record


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PhishGuard AI — Cloud Phishing & Scam Detector</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f4f6f9;
      color: #111827;
      line-height: 1.5;
    }
    header {
      background: #ffffff;
      border-bottom: 1px solid #e5e7eb;
      padding: 14px 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .brand {
      font-size: 18px;
      font-weight: 700;
      color: #111827;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .brand span {
      background: #1f4e79;
      color: #ffffff;
      font-size: 12px;
      padding: 2px 8px;
      border-radius: 999px;
      font-weight: 600;
    }
    nav { display: flex; gap: 20px; }
    nav button {
      background: none;
      border: none;
      font-size: 14px;
      font-weight: 600;
      color: #4b5563;
      cursor: pointer;
      padding: 4px 2px;
      border-bottom: 2px solid transparent;
    }
    nav button.active {
      color: #2563eb;
      border-bottom-color: #2563eb;
    }
    main {
      max-width: 820px;
      margin: 28px auto;
      padding: 0 20px 40px;
    }
    h1 { font-size: 24px; font-weight: 700; margin-bottom: 6px; }
    .subtitle { color: #4b5563; font-size: 14px; margin-bottom: 20px; }
    .card {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 18px;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }
    .samples-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 14px;
    }
    .pill {
      background: #eff6ff;
      color: #1d4ed8;
      border: 1px solid #bfdbfe;
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
    }
    .pill:hover { background: #dbeafe; }
    textarea, input[type="password"] {
      width: 100%;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      padding: 12px;
      font-size: 14px;
      font-family: inherit;
      outline: none;
    }
    textarea:focus, input[type="password"]:focus { border-color: #2563eb; }
    textarea { min-height: 110px; resize: vertical; margin-bottom: 12px; }
    .actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }
    .btn-primary {
      background: #2563eb;
      color: #ffffff;
      border: none;
      border-radius: 8px;
      padding: 10px 20px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
    }
    .btn-primary:hover { background: #1d4ed8; }
    .btn-primary:disabled { opacity: 0.6; cursor: wait; }
    .btn-secondary {
      background: #f3f4f6;
      color: #374151;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
    }
    .status-note { font-size: 12px; color: #6b7280; }
    .badge {
      display: inline-block;
      padding: 4px 12px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 700;
    }
    .badge-high { background: #fee2e2; color: #b91c1c; }
    .badge-mid { background: #fef3c7; color: #b45309; }
    .badge-safe { background: #dcfce7; color: #15803d; }
    .meter-wrap {
      height: 10px;
      background: #e5e7eb;
      border-radius: 999px;
      overflow: hidden;
      margin: 12px 0;
    }
    .meter-fill { height: 100%; transition: width 0.3s ease; }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin: 12px 0;
      font-size: 13px;
      color: #374151;
    }
    .explanation-box {
      background: #f8fafc;
      border-left: 3px solid #2563eb;
      padding: 10px 12px;
      font-size: 14px;
      color: #1f2937;
      margin: 10px 0;
      border-radius: 4px;
    }
    .flags-list { margin-left: 18px; font-size: 14px; color: #374151; }
    .flags-list li { margin-bottom: 4px; }
    .hidden { display: none; }
    .history-item {
      border-bottom: 1px solid #e5e7eb;
      padding: 12px 0;
    }
    .history-item:last-child { border-bottom: none; }
    footer {
      text-align: center;
      color: #6b7280;
      font-size: 12px;
      margin-top: 24px;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">PhishGuard AI <span>Gemini Cloud</span></div>
    <nav>
      <button id="tab-scanner" class="active" onclick="switchTab('scanner')">Scanner</button>
      <button id="tab-history" onclick="switchTab('history')">Scan History (<span id="hist-count">0</span>)</button>
      <button id="tab-settings" onclick="switchTab('settings')">Cloud API Setup</button>
    </nav>
  </header>

  <main>
    <section id="view-scanner">
      <h1>Phishing &amp; Scam Message Detector</h1>
      <p class="subtitle">Paste a suspicious email, SMS, or URL — or click a sample below — to classify it using the Google Gemini Cloud API.</p>

      <div class="card">
        <div class="samples-row" id="samples-container"></div>
        <textarea id="msg-input" placeholder="Paste suspicious SMS, email text, or URL here..."></textarea>
        <div class="actions">
          <div>
            <button class="btn-primary" id="analyze-btn" onclick="analyzeMessage()">Analyze Message</button>
            <button class="btn-secondary" onclick="clearInput()">Clear</button>
          </div>
          <span class="status-note" id="cloud-status">
            {% if env_key_set %}Cloud Status: GEMINI_API_KEY configured{% else %}Cloud Status: Set GEMINI_API_KEY in environment or Cloud API Setup tab{% endif %}
          </span>
        </div>
      </div>

      <div class="card hidden" id="result-card">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
          <span id="res-badge" class="badge"></span>
          <strong id="res-score" style="font-size:16px;"></strong>
        </div>
        <div class="meter-wrap"><div id="res-meter" class="meter-fill"></div></div>
        <div class="meta-grid">
          <div><strong>Cloud Service:</strong> <span id="res-source"></span></div>
          <div><strong>AI Model:</strong> <span id="res-model"></span></div>
        </div>
        <div id="res-explanation" class="explanation-box hidden"></div>
        <div style="margin-top:10px;">
          <strong style="font-size:14px;">Detected Red Flags:</strong>
          <ul id="res-flags" class="flags-list" style="margin-top:6px;"></ul>
        </div>
      </div>
    </section>

    <section id="view-history" class="hidden">
      <h1>Scan History</h1>
      <p class="subtitle">Recent scans logged during this session (also saved to <code>scan_results.json</code> when running locally).</p>
      <div class="card" id="history-container">
        <p class="status-note">No messages scanned yet.</p>
      </div>
    </section>

    <section id="view-settings" class="hidden">
      <h1>Cloud API Configuration</h1>
      <p class="subtitle">Optional: paste your Google Gemini API key here if you did not set <code>GEMINI_API_KEY</code> in environment variables.</p>
      <div class="card">
        <label for="token-input" style="display:block; font-size:13px; font-weight:600; margin-bottom:6px;">Google Gemini API Key</label>
        <input id="token-input" type="password" placeholder="Paste your Google AI Studio API key here" />
        <p class="status-note" style="margin-top:8px;">Used only for live HTTPS calls to the Google Gemini Developer API (<code>generativelanguage.googleapis.com</code>).</p>
      </div>
    </section>

    <footer>
      PhishGuard AI — Shreekar V J (ENG24CS0220) • Powered by Google Gemini API
    </footer>
  </main>

  <script>
    async function loadSamples() {
      const res = await fetch('/api/samples');
      const samples = await res.json();
      const container = document.getElementById('samples-container');
      container.innerHTML = '';
      samples.forEach(s => {
        const btn = document.createElement('button');
        btn.className = 'pill';
        btn.textContent = s.category;
        btn.onclick = () => {
          document.getElementById('msg-input').value = s.message;
        };
        container.appendChild(btn);
      });
    }

    function switchTab(tab) {
      ['scanner', 'history', 'settings'].forEach(t => {
        document.getElementById('view-' + t).classList.toggle('hidden', t !== tab);
        document.getElementById('tab-' + t).classList.toggle('active', t === tab);
      });
      if (tab === 'history') loadHistory();
    }

    function clearInput() {
      document.getElementById('msg-input').value = '';
      document.getElementById('result-card').classList.add('hidden');
    }

    function badgeClass(verdict) {
      if (verdict.includes('HIGH')) return 'badge badge-high';
      if (verdict.includes('SUSPICIOUS')) return 'badge badge-mid';
      return 'badge badge-safe';
    }

    function meterColor(verdict) {
      if (verdict.includes('HIGH')) return '#dc2626';
      if (verdict.includes('SUSPICIOUS')) return '#d97706';
      return '#16a34a';
    }

    async function analyzeMessage() {
      const text = document.getElementById('msg-input').value.trim();
      if (!text) return alert('Please paste or select a message first.');
      const apiKey = document.getElementById('token-input').value.trim();
      const btn = document.getElementById('analyze-btn');
      btn.disabled = true;
      btn.textContent = 'Analyzing...';

      try {
        const res = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, api_key: apiKey })
        });
        const data = await res.json();
        document.getElementById('result-card').classList.remove('hidden');
        const badge = document.getElementById('res-badge');
        badge.className = badgeClass(data.verdict);
        badge.textContent = data.verdict;
        document.getElementById('res-score').textContent = data.phishing_probability_pct + '% Phishing Risk';
        const meter = document.getElementById('res-meter');
        meter.style.width = Math.max(data.phishing_probability_pct, 4) + '%';
        meter.style.background = meterColor(data.verdict);
        document.getElementById('res-source').textContent = data.inference_source;
        document.getElementById('res-model').textContent = data.model;

        const expBox = document.getElementById('res-explanation');
        if (data.explanation) {
          expBox.textContent = 'AI Analysis: ' + data.explanation;
          expBox.classList.remove('hidden');
        } else {
          expBox.classList.add('hidden');
        }

        const ul = document.getElementById('res-flags');
        ul.innerHTML = '';
        if (data.red_flags.length === 0) {
          ul.innerHTML = '<li>No red flags detected.</li>';
        } else {
          data.red_flags.forEach(f => {
            const li = document.createElement('li');
            li.textContent = f;
            ul.appendChild(li);
          });
        }
        loadHistory();
      } catch (err) {
        alert('Error analyzing message: ' + err);
      } finally {
        btn.disabled = false;
        btn.textContent = 'Analyze Message';
      }
    }

    async function loadHistory() {
      const res = await fetch('/api/history');
      const items = await res.json();
      document.getElementById('hist-count').textContent = items.length;
      const box = document.getElementById('history-container');
      if (!items.length) {
        box.innerHTML = '<p class="status-note">No messages scanned yet.</p>';
        return;
      }
      box.innerHTML = items.map(it => `
        <div class="history-item">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span class="${badgeClass(it.verdict)}">${it.verdict} (${it.phishing_probability_pct}%)</span>
            <span class="status-note">${it.timestamp}</span>
          </div>
          <div style="font-size:14px; margin-bottom:4px;">${it.message_preview}</div>
          <div class="status-note">${it.inference_source} • ${it.model}</div>
        </div>
      `).join('');
    }

    loadSamples();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, env_key_set=bool(os.environ.get("GEMINI_API_KEY")))


@app.route("/api/samples")
def api_samples():
    return jsonify(load_samples())


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    body = request.get_json(silent=True) or {}
    text = str(body.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Message text is required."}), 400
    custom_key = body.get("api_key") or body.get("token")
    result = analyze_message(text, custom_key=custom_key)
    return jsonify(result)


@app.route("/api/history")
def api_history():
    return jsonify(SCAN_HISTORY)


if __name__ == "__main__":
    print("Starting PhishGuard AI Web App at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
