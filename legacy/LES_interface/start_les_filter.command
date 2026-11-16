#!/bin/bash
# Double-click this file or run: ./start_les_filter.sh

cd "$(dirname "$0")"

echo "Starting LES Filter server..."
echo "Browser will open automatically."
echo ""

# Open browser after a short delay (gives server time to start)
(sleep 3 && open "http://localhost:8080" 2>/dev/null || xdg-open "http://localhost:8080" 2>/dev/null) &

# Start Julia server with all available threads
julia -t auto server.jl