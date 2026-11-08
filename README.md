# Alzheimer's SNN Patient Monitoring Dashboard

A Streamlit research prototype for preprocessed EEG classification and separate manual physiological monitoring.

## Model files required

The source ZIP does **not** contain your trained model artifacts. Copy these three files from your Google Drive/Colab training output into `models/`:

- `alzheimer_snn_best.pth`
- `feature_scaler.pkl`
- `model_config.json`

The app can launch without them, but EEG inference will be unavailable until they are added.

## Folder structure

```text
Alzheimers_SNN_Dashboard/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   ├── alzheimer_snn_best.pth
│   ├── feature_scaler.pkl
│   └── model_config.json
└── utils/
    ├── __init__.py
    ├── snn_model.py
    ├── eeg_processing.py
    └── monitoring.py
```

## Run on Windows

Open PowerShell in this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Open the local address printed by Streamlit (usually `http://localhost:8501`).

## EEG upload format

The `.npy` upload must have shape `(number_of_windows, 19, 1000)`, sampled at 500 Hz, and be preprocessed consistently with model training.

## Upload to GitHub

1. Create a repository on GitHub.
2. Upload the project files and folders.
3. Do not upload `.venv/`, raw EEG data, or identifiable patient information.
4. Add trained model files only if you are allowed to share them. For large model files, consider Git LFS or private/secure storage.

Git commands:

```bash
git init
git add .
git commit -m "Initial Alzheimer's SNN dashboard"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

## Streamlit Community Cloud

Deploy the GitHub repository with `app.py` as the main file. Make sure the trained model files are available to the deployed app. Do not commit real patient EEG or personal health information to a public repository.

## Current prototype 

- Heart rate and GSR are manually entered; physical sensors are not connected.
- Restroom reminders use manually entered elapsed time.
- EEG output is a research result, not a clinical diagnosis.
- EEG and monitoring data are not synchronized unless collected together.
- A mixed file containing different participants is only suitable for software testing, not single-patient assessment.
