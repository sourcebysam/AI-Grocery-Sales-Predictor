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

📁 Project Structure
AI-Grocery-Sales-Predictor/
├─ data/
│  └─ sample_sales.csv
├─ src/
│  ├─ app.py
│  ├─ predictor.py
│  └─ visualize.py
├─ .gitignore
├─ requirements.txt
├─ README.md
├─ LICENSE

📦 Data Format

Your CSV must contain at least these columns:

Date

Units_Sold

2025-01-01

20

2025-01-02

25

Optional columns that will be shown in analytics if present: Item, Price, Profit.

🚀 Features

Upload CSV and preview the data

Auto-cleaning for common date/number issues

Choose forecast horizon (7–120 days)

Prophet-based forecast with trend/seasonality components

Interactive charts (historical + forecast with confidence intervals)

Download predictions as CSV

Fallback to Linear Regression if Prophet isn’t available

🧠 Tech Stack

Python 3.9+

Streamlit, Pandas, Prophet, scikit-learn, Matplotlib

☁️ Deploy

Streamlit Community Cloud (recommended)

Push this repo to GitHub.

Go to share.streamlit.io → New app → select your repo → main file src/app.py.

Add any secrets (not required for this app) and deploy.

Hugging Face Spaces (Gradio/Streamlit)

Create a new Space → Streamlit template.

Upload this repo’s files.

Set App File to src/app.py and Python version in Space settings if needed.

🧪 Sample Usage

Use the included data/sample_sales.csv to test quickly.

Try adding more rows, multiple months, or different items.

