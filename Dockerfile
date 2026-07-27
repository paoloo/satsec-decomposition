FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace/satsec-decomposition
COPY requirements-lock.txt pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir -r requirements-lock.txt && \
    python -m pip install --no-cache-dir --no-deps -e .
COPY . .
ENV PYTHONPATH=/workspace/satsec-decomposition/src
CMD ["python", "tools/audit_dataset.py", "--data", "data/tuning_set.v2.jsonl"]

