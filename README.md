# 🛒 AI Grocery Store Sales Predictor (Interactive)

An open-source **Python + AI** web app that forecasts grocery store sales from your historical data.

- **Streamlit UI** for a user-friendly experience
- **Prophet** for robust time-series forecasting
- **Interactive charts** and **downloadable predictions**
- **Upload your CSV** (Date, Units_Sold) and get **30–90 day forecasts**

---

## ✨ Demo (Run Locally)
```bash
# 1) Clone
git clone https://github.com/<your-username>/AI-Grocery-Sales-Predictor.git
cd AI-Grocery-Sales-Predictor

# 2) Create & activate a virtual environment (optional but recommended)
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run the Streamlit app
streamlit run src/app.py