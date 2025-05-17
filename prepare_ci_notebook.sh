name: Test Notebook with repo2docker

on:
  push:
  pull_request:

jobs:
  repo2docker-test:
    runs-on: ubuntu-latest
    steps:
      - name: Check out code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.x'

      - name: Install repo2docker & Docker client
        run: |
          python -m pip install --upgrade pip
          pip install jupyter-repo2docker docker

      - name: Build Docker image
        run: |
          repo2docker --no-run --image-name notebook-test .

      - name: Execute notebook inside container
        run: |
          # Check if CI notebook config exists
          if [ -f .ci_notebook_config ]; then
            source .ci_notebook_config
            echo "Using CI notebook: $NOTEBOOK"
          else
            # Default notebook name if no config
            NOTEBOOK="MLFLOW.ipynb"
            echo "Using default notebook: $NOTEBOOK"
          fi
          
          # launch in background
          docker run --name nbtest -d notebook-test tail -f /dev/null

          # run & fail fast if any cell errors
          docker exec nbtest \
            jupyter nbconvert \
              --to notebook \
              --execute $NOTEBOOK \
              --output executed.ipynb \
              --ExecutePreprocessor.timeout=600

          # tear down
          docker stop nbtest