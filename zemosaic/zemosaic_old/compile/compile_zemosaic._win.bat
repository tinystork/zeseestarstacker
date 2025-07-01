@echo off
REM === Script de compilation ZeMosaic (.exe) ===
REM === ZeMosaic Build Script (.exe) ===

REM Ce script doit être lancé depuis le dossier racine du projet
REM This script must be run from the project root folder

REM Active l’environnement virtuel local
REM Activate the local virtual environment
call .venv\Scripts\activate

REM Lancer PyInstaller avec le fichier .spec
REM Launch PyInstaller using the .spec file
pyinstaller zemosaic.spec

REM Pause pour voir les messages en fin de compilation
REM Pause to view final messages after build
pause
