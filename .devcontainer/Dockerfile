FROM ghcr.io/i4ds/karabo-pipeline:latest

RUN apt update && apt install -y build-essential
RUN pip install black mypy pylint pydocstyle pycln isort pyupgrade pre-commit pytest ipykernel katdal flake8
