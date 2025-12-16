#!/bin/bash
# Script to open xschem symbols in xschem using docker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"

# Check if docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed or not in PATH"
    exit 1
fi

# Check if output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory $OUTPUT_DIR does not exist"
    echo "Please run xschem_bridge_example.py first to generate symbols"
    exit 1
fi

# Check if symbols exist
SYM_COUNT=$(find "$OUTPUT_DIR" -name "*.sym" | wc -l)
if [ "$SYM_COUNT" -eq 0 ]; then
    echo "Error: No .sym files found in $OUTPUT_DIR"
    echo "Please run xschem_bridge_example.py first to generate symbols"
    exit 1
fi

echo "Opening xschem with generated symbols..."
echo "Symbol directory: $OUTPUT_DIR"
echo ""

# Get current user ID to avoid permission issues
UID=$(id -u)
GID=$(id -g)

# Run xschem in docker
# Mount the output directory so xschem can access the symbols
# Note: The docker image may have a custom entrypoint, so we use bash -c
docker run --rm -it \
    -v "$OUTPUT_DIR:/workspace/symbols" \
    -v "$SCRIPT_DIR:/workspace" \
    -w /workspace \
    --user "$UID:$GID" \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --entrypoint="" \
    hpretl/iic-osic-tools \
    bash -c "xschem || (echo 'Note: xschem may need to be launched differently in this docker image' && echo 'Symbols are in /workspace/symbols/' && bash)"

