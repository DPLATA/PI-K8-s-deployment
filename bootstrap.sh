#!/bin/sh
export FLASK_APP=app/app.py
export FLASK_ENV=development
#source $(pipenv --venv)/bin/activate
flask run
