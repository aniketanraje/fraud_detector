#!/usr/bin/env bash

set -e  # exit on any error

echo "🚀 Bootstrapping Fraud Detection System..."

# --- Check Python ---
if ! command -v python3 &> /dev/null
then
    echo "❌ Python3 is required but not installed."
    exit 1
fi

# --- Create virtual environment ---
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# --- Upgrade pip ---
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# --- Install PyTorch (CPU version) ---
echo "🧠 Installing PyTorch..."
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

# --- Install dependencies ---
echo "📚 Installing project dependencies..."
pip install -r requirements.txt

# --- Train model if not present ---
if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
    echo "🏋️ Training model (first-time setup)..."
    python -m src.main --mode train
else
    echo "✅ Model already exists. Skipping training."
fi

# --- Run Streamlit ---
echo "🌐 Launching Streamlit app..."
python -m streamlit run app/main_ui.p