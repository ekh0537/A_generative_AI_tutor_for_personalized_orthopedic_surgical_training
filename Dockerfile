FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .
ENTRYPOINT ["ag-tri-ct"]
