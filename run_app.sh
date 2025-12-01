#!/bin/bash

# People Counter AI - Unix/Linux/Mac Launcher
# Quick start script for the Streamlit application

clear

echo "============================================================"
echo " 👥 PEOPLE COUNTER AI - PRESENTATION MODE"
echo "============================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found"
    exit 1
fi

echo "📦 Installing dependencies..."
python3 -m pip install -r requirements.txt -q

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install dependencies"
    exit 1
fi

echo ""
echo "============================================================"
echo "🚀 Launching People Counter AI..."
echo ""
echo "🌐 Web App URL: http://localhost:8501"
echo "💡 Press Ctrl+C to stop the server"
echo "============================================================"
echo ""

python3 -m streamlit run streamlit/app.py --logger.level=info

echo ""
echo "✓ Application stopped"
