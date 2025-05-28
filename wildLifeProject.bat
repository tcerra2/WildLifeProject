@echo off
cd /d "C:\Users\tcerr\Documents\wildlifeproject"
call venv\Scripts\activate
cd "wildlife_detector_starter_repo_integrated (1)\app"
streamlit run main.py
pause
