#!/usr/bin/env bash
# CI guard: no escape-hatch APIs may leak into production.
#
# `#[doc(hidden)]` and the `tag_as_inertial!` macro are the project's two
# explicit escape-hatch markers — anything that's deliberately not part of
# the public API surface but stays callable. Outside of legitimate
# typed-construction primitives at module boundaries (e.g.,
# `from_dvec3_unchecked`, which uses `_unchecked` per established Rust
# convention rather than `#[doc(hidden)]`), neither marker should appear in
# `crates/` or `src/`.
#
# Allowed exceptions: lines containing `// allowed:` are exempt. Use
# sparingly; document each exemption in the PR description.
set -euo pipefail

matches=$(grep -rEn '#\[doc\(hidden\)\]|tag_as_inertial!' crates/ src/ \
  | grep -v '// allowed:' || true)

if [ -n "$matches" ]; then
    echo "FAIL: escape-hatch markers detected" >&2
    echo "$matches" >&2
    exit 1
fi

echo "OK: no escape-hatch markers"
