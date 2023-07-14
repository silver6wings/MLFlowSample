# docker build -t my_mlflow .
# docker build -f Dockerfile-as-root --progress plain --no-cache -t mlflow_tracker_slim_as_root .
# docker run --name my_mlflow_instance -p 5005:5000 -d my_mlflow


# Good practice: Use official base images
FROM python:3.10-slim

# Good practice: upgrade distro packages (with last security patches).
RUN apt-get update && apt-get -y upgrade \
    && pip install --upgrade pip \
    && pip --version

RUN apt-get update && apt-get install -y procps \
    && rm -rf /var/lib/apt/lists/*

# Install mlflow dependencies:
WORKDIR /mlflow/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

# Expose mlflow ports
EXPOSE 5000

# Launch the mlflow server
CMD mlflow server --backend-store-uri ${BACKEND_STORE_URI} \
                  --default-artifact-root ${DEFAULT_ARTIFACT_ROOT} \
                  --artifacts-destination ${DEFAULT_ARTIFACTS_DESTINATION} \
                  --no-serve-artifacts \
                  --host 0.0.0.0 --port 5000
