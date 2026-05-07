#!/usr/bin/env bash
# Install the CycloneDX/sbom-utility binary.
#
# AIkaBoOM uses sbom-utility (https://github.com/CycloneDX/sbom-utility) for
# CycloneDX 1.7 BOM validation. The binary is optional at runtime: if it's
# not on PATH, validate_cyclonedx() returns valid=None / validator="skipped"
# so the rest of the pipeline still runs.
#
# This script:
#   1. Detects the platform (Linux/macOS, x86_64/arm64)
#   2. Downloads the latest matching release tarball
#   3. Extracts the binary into ~/.local/bin (creating it if needed)
#   4. Prints the install path so the user can confirm
#
# Usage:
#   bash tools/install_sbom_utility.sh
#   bash tools/install_sbom_utility.sh --version v0.18.0   # pin a specific release

set -euo pipefail

VERSION="latest"
INSTALL_DIR="${SBOM_UTILITY_INSTALL_DIR:-$HOME/.local/bin}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,/^set -euo/p' "$0" | grep -E '^# ' | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

# Detect platform
uname_s="$(uname -s)"
uname_m="$(uname -m)"

case "$uname_s" in
    Linux)  os="Linux" ;;
    Darwin) os="Darwin" ;;
    *)
        echo "Unsupported OS: $uname_s. Install manually from https://github.com/CycloneDX/sbom-utility/releases." >&2
        exit 3
        ;;
esac

case "$uname_m" in
    x86_64|amd64) arch="amd64" ;;
    arm64|aarch64) arch="arm64" ;;
    *)
        echo "Unsupported arch: $uname_m. Install manually." >&2
        exit 3
        ;;
esac

# Resolve the latest tag if not pinned
if [[ "$VERSION" == "latest" ]]; then
    VERSION=$(curl -fsSL https://api.github.com/repos/CycloneDX/sbom-utility/releases/latest \
        | grep -oE '"tag_name"\s*:\s*"[^"]+"' \
        | head -n1 \
        | sed -E 's/.*"([^"]+)"$/\1/')
    if [[ -z "$VERSION" ]]; then
        echo "Could not resolve latest version. Pass --version explicitly." >&2
        exit 4
    fi
fi

# Tarball name pattern: sbom-utility-<version>-<os>-<arch>.tar.gz
# (Release tags are like v0.18.0; the asset name strips the leading 'v'.)
asset_version="${VERSION#v}"
asset="sbom-utility-${asset_version}-${os}-${arch}.tar.gz"
url="https://github.com/CycloneDX/sbom-utility/releases/download/${VERSION}/${asset}"

mkdir -p "$INSTALL_DIR"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

echo "Fetching $url ..."
curl -fSL "$url" -o "$tmpdir/sbom-utility.tar.gz"
tar -xzf "$tmpdir/sbom-utility.tar.gz" -C "$tmpdir"

# The tarball typically contains 'sbom-utility' at the top level; copy it.
binary_src=$(find "$tmpdir" -maxdepth 2 -type f -name 'sbom-utility' -perm -u+x | head -n1)
if [[ -z "$binary_src" ]]; then
    echo "Could not find sbom-utility binary in tarball." >&2
    exit 5
fi

install -m 0755 "$binary_src" "$INSTALL_DIR/sbom-utility"

echo "Installed: $INSTALL_DIR/sbom-utility"
"$INSTALL_DIR/sbom-utility" --version || true

# Hint if the install dir isn't on PATH
case ":$PATH:" in
    *":$INSTALL_DIR:"*) ;;
    *)
        echo
        echo "Note: $INSTALL_DIR is not on your PATH."
        echo "Add to your shell config:  export PATH=\"$INSTALL_DIR:\$PATH\""
        ;;
esac
