#!/bin/sh

nohup redis-server &

python3 ingest.py

uvicorn "main:app" --port 7860 --host 0.0.0.0
