#!/usr/bin/env bash
# The verl every step runs: PEAR's checkout (third_party/pear/3rdparty/verl/src, its pinned commit with PEAR's patches)
# with this repository's patches (patches/verl) on top, one commit each, checked by PEAR's own patched_checkout.sh over
# both series.
#   bash scripts/verl.sh           applies the patches of patches/verl the checkout lacks (scripts/setup.sh), then checks
#   bash scripts/verl.sh --check   checks only (every step, before it runs)
# VERL_CHANGE_IN_PROGRESS=1 accepts a checkout that differs, as PEAR's check does.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VERL=$ROOT/third_party/pear/3rdparty/verl
# shellcheck source=../third_party/pear/3rdparty/verl/UPSTREAM
source "$VERL/UPSTREAM"
PROJECT=verl
UPSTREAM_URL=$VERL_URL
UPSTREAM_COMMIT=$VERL_COMMIT
CHECKOUT_DIR=$VERL/src
CHANGE_IN_PROGRESS=${VERL_CHANGE_IN_PROGRESS:-}
# both series in one directory, PEAR's first (the names sort so)
PATCHES_DIR=$(mktemp -d)
ln -s "$VERL"/patches/*.patch "$ROOT"/patches/verl/*.patch "$PATCHES_DIR"
# shellcheck source=../third_party/pear/3rdparty/patched_checkout.sh
source "$VERL/../patched_checkout.sh"

if [ "${1:-}" != --check ] && [ -e "$CHECKOUT_DIR" ]; then
  applied=$(patch_ids_of_checkout)
  for patch in "$ROOT"/patches/verl/*.patch; do
    if ! grep -qx "$(git patch-id --stable < "$patch" | cut -d ' ' -f 1)" <<< "$applied"; then
      git -C "$CHECKOUT_DIR" -c user.name="$PATCH_AUTHOR_NAME" -c user.email="$PATCH_AUTHOR_EMAIL" \
        am -q --keep-non-patch "$patch"
    fi
  done
fi
status=0
checkout "${1:-}" || status=$?
rm -rf "$PATCHES_DIR"
exit "$status"
