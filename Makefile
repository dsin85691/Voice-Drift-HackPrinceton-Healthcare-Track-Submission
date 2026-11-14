.PHONY: run test seed artifacts mock budget

run:        ## serve the app on 127.0.0.1:8000
	python -m uvicorn server.app:app --host 127.0.0.1 --port 8000

mock:       ## fixture API for frontend dev
	python -m uvicorn server.mock_server:app --host 127.0.0.1 --port 8001

test:
	python -m pytest -q

seed:       ## regenerate simulated scenario JSON
	python -m server.seed.make_seed

budget:     ## GPU hours used / remaining (set DRIVE to the Colab Drive dir)
	python tools/budget.py --drive $(DRIVE)

artifacts:  ## pull the private model bundle (set ARTIFACT_URL to a zip on Drive / private store)
	python tools/fetch_artifacts.py $(ARTIFACT_URL)
