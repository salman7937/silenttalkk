# SilentTalk — Project Setup Guide

## ✅ Requirements (پہلے یہ install ہونے چاہئیں)
- [Python 3.11](https://www.python.org/downloads/release/python-3119/) — install کرتے وقت **"Add to PATH"** ضرور check کریں
- [Node.js](https://nodejs.org/) — LTS version download کریں

---

## 🚀 Setup (صرف ایک بار کریں)

### Step 1 — Project folder میں جائیں
```bash
cd silenttalkk
```

### Step 2 — Python Virtual Environment بنائیں اور سب libraries install کریں
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3 — Frontend libraries install کریں
```bash
npm install
```

---

## ▶️ Project چلانے کے لیے (ہر بار)

**Terminal 1 — Backend (Flask):**
```bash
.venv\Scripts\activate
cd backend
python app.py
```

**Terminal 2 — Frontend (React):**
```bash
npm start
```

---

## ⚠️ نوٹ
- Backend اور Frontend دونوں **الگ الگ terminals** میں چلائیں
- Backend پہلے start کریں، پھر Frontend
