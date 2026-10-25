# Query API and proposal-context image for compose.yaml. Data and secrets are mounted at runtime.
FROM python:3.13-slim-bookworm
RUN pip install --no-cache-dir pillow==12.3.0 pillow-heif==1.8.1 numpy==2.5.4 scipy==1.18.1 py360convert==1.0.4
WORKDIR /app
COPY --chmod=0755 explore/object_api.py explore/proposal_context.py explore/
# Mirror the host's data symlinks so resolved paths match absolute paths stored in result.json.
RUN ln -s /mnt/data/projects/typology-machine/data /app/data \
 && mkdir -p /home/mateo/projects/typology-machine \
 && ln -s /mnt/data/projects/typology-machine/data /home/mateo/projects/typology-machine/data
USER 1000:1000
ENV PYTHONUNBUFFERED=1
