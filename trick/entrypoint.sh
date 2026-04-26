#!/bin/bash
# Wrapper invoked when the jeod-trick image is run without overriding CMD.
# Prints a clear usage message if generate_references.sh wasn't bind-mounted,
# otherwise delegates to it.
#
# Replaces the inline `printf '...' > /entrypoint.sh && chmod +x` build-step
# that produced a syntax-broken file (#86); copying the script verbatim via
# Dockerfile COPY keeps the source readable and avoids escape-soup.

set -euo pipefail

if [ ! -f /generate_references.sh ]; then
    cat >&2 <<'USAGE'
Error: /generate_references.sh not found in the container.

The script must be bind-mounted at runtime. From the bevy_jeod root:

  docker run --rm \
    -v "$(pwd)/test_data:/output" \
    -v "$(pwd)/trick/generate_references.sh:/generate_references.sh:ro" \
    jeod-trick

Or, more conveniently, run via the xtask wrapper:

  cargo xtask regenerate-tier3            # incremental
  cargo xtask regenerate-tier3 --force    # regenerate everything

USAGE
    exit 1
fi

exec /bin/bash /generate_references.sh "$@"
