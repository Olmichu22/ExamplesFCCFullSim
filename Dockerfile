# FCC Event Display — Docker image
#
# Base: key4hep-stack (provides ROOT, podio, k4geo)
# Adds: dash, plotly, dash-bootstrap-components
#
# Build:
#   docker build -t fcc-event-display .
#
# Run (see run_display.sh):
#   docker run --rm -p 8050:8050 \
#     -v /path/to/file.root:/data/input.root:ro \
#     fcc-event-display -i /data/input.root

FROM ghcr.io/key4hep/key4hep-stack:2024-10-03

# Install Python UI dependencies (key4hep already provides ROOT/podio/numpy)
RUN pip install --no-cache-dir \
    "dash==4.1.0" \
    "dash-bootstrap-components==2.0.4" \
    "plotly==6.7.0"

WORKDIR /app

# Copy EventDisplay package and local modules
COPY EventDisplay/ /app/EventDisplay/
COPY modules/      /app/modules/

# Make project root importable so EventDisplay can find modules/
ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8050

ENTRYPOINT ["python", "/app/EventDisplay/event_display_dash.py"]
