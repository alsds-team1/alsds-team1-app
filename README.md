# AI-Assisted Location Decision Support System (ALSDS)
ALY6080: Integrated Experiential Learning 

Team 1: Jirapa, Zexuan, and Yixuan

Web direction: https://alsds-team1-app-e3fed9azh2fpb4ej.eastus-01.azurewebsites.net/

## Version 2.0



## Version 1.5


## Version 1.0





This repository provides the **baseline application and infrastructure** for the ALSDS capstone project.

It includes:

- Core **data files** for Worcester, MA (CBGs, POIs, visits, parameters)
- A **baseline Huff-style model implementation** (`huff_engine.py`)
- A **Flask-based web application** with:
  - Guided chatbot (Azure OpenAI / GPT-4o)
  - Interactive map (Leaflet + GeoJSON)
  - Model execution API
- Azure deployment configuration (App Service compatible)


## Baseline Version


Stable infrastructure release: v1-baseline


All teams must start from this version before making any changes.



## Critical Infrastructure

Unchanged files:

startup.sh
requirements.txt (do not remove any current requirements, you can add more items if required.)
app.py (initially, unless explicitly instructed)
deployment configuration (GitHub Actions / Azure settings)
environment variable naming
API route structure (/api/run_huff, /api/ask)


## Modified parts


- huff_engine.py
- database design (Azure SQL)
- data preprocessing / optimization
- optional UI enhancements

