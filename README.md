# CS178 Final Project

## 1. Create a Virtual Environment
# Run the following command to create a virtual environment named .venv:
python -m venv .venv

## 2. Activate the Virtual Environment
# Activate the virtual environment:
# - On macOS/Linux:
source .venv/bin/activate
# - On Windows:
.venv\Scripts\activate

## 3. Install Required Packages
# Use the requirements.txt file to install all necessary packages:
pip install -r requirements.txt
pip install notebook ipykernel
    also create a notebook enviroment thing:
    python -m ipykernel install --name .venv --display-name "cs178 (.venv)"

## 4. Deactivate the Virtual Environment (Optional)
# When you're done working, you can deactivate the environment:
deactivate
