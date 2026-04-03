#!/bin/bash
set -euo pipefail

# ECHELON Setup
# Usage
#   ./setup.sh          - pip install + ROMs only (local dev)
#   ./setup.sh --vast   - full Vast.ai provisioning (system deps + venv + install)

if [[ "${1:-}" == "--vast" ]]; then
    echo "=== Vast.ai full setup ==="

    apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        libgl1-mesa-glx mesa-utils \
        unrar wget git ffmpeg \
        libosmesa6-dev libglew-dev patchelf

    python3.12 -m venv /workspace/venv
    source /workspace/venv/bin/activate
    pip install --upgrade pip setuptools wheel
fi

pip install -r requirements.txt
AutoROM --accept-license

echo "=== Setup complete ==="
