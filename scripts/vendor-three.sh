#!/usr/bin/env bash
# Re-vendor three.js into web/vendor/three (no CDN needed at runtime).
#   scripts/vendor-three.sh [version]
set -euo pipefail
VERSION="${1:-0.170.0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/web/vendor/three"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

(cd "$TMP" && npm pack "three@$VERSION" --silent >/dev/null && tar xzf "three-$VERSION.tgz")
PKG="$TMP/package"

rm -rf "$DEST"
mkdir -p "$DEST/build" "$DEST/examples/jsm/loaders" "$DEST/examples/jsm/utils" "$DEST/examples/jsm/environments"
cp "$PKG/LICENSE" "$DEST/LICENSE"
cp "$PKG/build/three.module.min.js" "$DEST/build/"
cp "$PKG/examples/jsm/loaders/GLTFLoader.js" "$DEST/examples/jsm/loaders/"
cp "$PKG/examples/jsm/utils/BufferGeometryUtils.js" "$DEST/examples/jsm/utils/"
cp "$PKG/examples/jsm/environments/RoomEnvironment.js" "$DEST/examples/jsm/environments/"
echo "$VERSION" > "$DEST/VERSION"
echo "Vendored three@$VERSION into $DEST"
