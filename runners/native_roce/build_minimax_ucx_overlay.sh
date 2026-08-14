#!/usr/bin/env bash
set -euo pipefail

# Build the native-RoCE UCX overlay already validated on NKX GB300, but bind
# it to the exact MiniMax image's NIXL plugin. Run once, in one writable
# container task; benchmark ranks only consume and verify the finished files.

readonly UCX_COMMIT="8a6b06fb880accbb933a79cda893883872c68d9d"
readonly UCX_VERSION="1.22.0"
readonly UCX_ARCHIVE_SHA256="0515b9d21188c88c65b92b67f73c7bf2e121d7b6ebe1931cc59aa3354447fae1"
readonly GPUNETIO_COMMIT="f44728fb023ba61911b705dba289b1650ce2eb28"
readonly GPUNETIO_ARCHIVE_SHA256="c5b747a29bdf943c5031470f514dbb88e3a4eacbd67066d71a78d7cdd8af4d32"
readonly PATCHELF_VERSION="0.18.0"
readonly PATCHELF_ARCHIVE_SHA256="ae13e2effe077e829be759182396b931d8f85cfb9cfe9d49385516ea367ef7b2"
readonly PATCHELF_SHA256="4bba32f55213f33e3f94a5ee7a298612c14a25b86f31a9f1aea763766277d4c6"
readonly NIXL_PLUGIN_SHA256="961cbe4378d11f99194da241c90c1e3e3f3304e40eede69ee4135e6009f58d9a"
readonly NIXL_SITE_PACKAGES="/usr/local/lib/python3.12/dist-packages"
readonly SOURCE_PLUGIN="${NIXL_SITE_PACKAGES}/.nixl_cu13.mesonpy.libs/plugins/libplugin_UCX.so"

if [[ $# -ne 2 ]]; then
    echo "usage: $0 OVERLAY_PREFIX DOWNLOAD_CACHE" >&2
    exit 2
fi

readonly overlay_prefix="$1"
readonly download_cache="$2"
readonly manifest="${overlay_prefix}/inferencex-ucx-overlay.manifest"
readonly checksums="${overlay_prefix}/inferencex-ucx-overlay.sha256"

download_pinned() {
    local url="$1" target="$2" expected="$3" actual temporary
    mkdir -p "$(dirname "$target")"
    if [[ -f "$target" ]]; then
        actual="$(sha256sum "$target" | awk '{print $1}')"
        [[ "$actual" == "$expected" ]] || {
            echo "checksum mismatch for existing download: $target" >&2
            exit 1
        }
        return
    fi
    temporary="${target}.incomplete-${SLURM_JOB_ID:-manual}"
    [[ ! -e "$temporary" ]] || {
        echo "stale download staging path exists: $temporary" >&2
        exit 1
    }
    curl -fL --retry 5 --output "$temporary" "$url"
    actual="$(sha256sum "$temporary" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || {
        echo "download checksum mismatch: url=$url expected=$expected actual=$actual" >&2
        exit 1
    }
    mv "$temporary" "$target"
}

validate_existing() {
    [[ -f "$manifest" && -f "$checksums" ]] || return 1
    grep -qx "ucx_commit=$UCX_COMMIT" "$manifest"
    grep -qx "ucx_version=$UCX_VERSION" "$manifest"
    grep -qx "gpunetio_commit=$GPUNETIO_COMMIT" "$manifest"
    grep -qx "nixl_plugin_sha256=$NIXL_PLUGIN_SHA256" "$manifest"
    (cd "$overlay_prefix" && sha256sum --check "$(basename "$checksums")")
}

mkdir -p "$(dirname "$overlay_prefix")" "$download_cache"
exec 9>"${overlay_prefix}.lock"
flock -w 600 9 || {
    echo "failed to acquire overlay build lock: ${overlay_prefix}.lock" >&2
    exit 1
}

if validate_existing; then
    echo "validated existing UCX overlay: $overlay_prefix"
    exit 0
fi
[[ ! -e "$overlay_prefix" ]] || {
    echo "overlay exists without a valid manifest: $overlay_prefix" >&2
    exit 1
}

readonly ucx_archive="${download_cache}/ucx-${UCX_COMMIT}.tar.gz"
readonly gpunetio_archive="${download_cache}/gpunetio-${GPUNETIO_COMMIT}.tar.gz"
readonly patchelf_archive="${download_cache}/patchelf-${PATCHELF_VERSION}-aarch64.tar.gz"

download_pinned \
    "https://github.com/openucx/ucx/archive/${UCX_COMMIT}.tar.gz" \
    "$ucx_archive" "$UCX_ARCHIVE_SHA256"
download_pinned \
    "https://github.com/NVIDIA-DOCA/gpunetio/archive/${GPUNETIO_COMMIT}.tar.gz" \
    "$gpunetio_archive" "$GPUNETIO_ARCHIVE_SHA256"
download_pinned \
    "https://github.com/NixOS/patchelf/releases/download/${PATCHELF_VERSION}/patchelf-${PATCHELF_VERSION}-aarch64.tar.gz" \
    "$patchelf_archive" "$PATCHELF_ARCHIVE_SHA256"

[[ "$(sha256sum "$SOURCE_PLUGIN" | awk '{print $1}')" == "$NIXL_PLUGIN_SHA256" ]] || {
    echo "the exact MiniMax NIXL UCX plugin is not present" >&2
    exit 1
}

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    autoconf automake cuda-nvml-dev-13-0 flex gcc g++ libibumad-dev \
    libibverbs-dev libnl-3-dev libnl-route-3-dev libnuma-dev libtool make \
    pkg-config

readonly work_root="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/inferencex-ucx.XXXXXX")"
readonly stage_prefix="${overlay_prefix}.incomplete-${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$work_root"' EXIT
[[ ! -e "$stage_prefix" ]] || {
    echo "overlay staging path exists: $stage_prefix" >&2
    exit 1
}

mkdir -p "$work_root/ucx" "$work_root/patchelf"
tar -xzf "$ucx_archive" --strip-components=1 -C "$work_root/ucx"
tar -xzf "$gpunetio_archive" --strip-components=1 \
    -C "$work_root/ucx/external/gpunetio"
tar -xzf "$patchelf_archive" -C "$work_root/patchelf"
readonly patchelf_bin="$work_root/patchelf/bin/patchelf"
[[ "$(sha256sum "$patchelf_bin" | awk '{print $1}')" == "$PATCHELF_SHA256" ]]

pushd "$work_root/ucx" >/dev/null
./autogen.sh
./contrib/configure-release-mt \
    --prefix="$stage_prefix" \
    --enable-shared --disable-static --disable-doxygen-doc \
    --enable-experimental-api --enable-optimizations --enable-cma \
    --enable-devel-headers --with-cuda=/usr/local/cuda --with-verbs \
    --with-dm --without-gdrcopy --without-efa --without-dc \
    --without-rdmacm --without-gga
make -j"$(nproc)"
make install-strip
popd >/dev/null

readonly original_nixl_libs="${NIXL_SITE_PACKAGES}/.nixl_cu13.mesonpy.libs"
mkdir -p "$stage_prefix/nixl/plugins"
install -m 0755 "$SOURCE_PLUGIN" "$stage_prefix/nixl/plugins/libplugin_UCX.so"
"$patchelf_bin" \
    --replace-needed libucp-fb7bfdea.so.0.0.0 libucp.so.0 \
    --replace-needed libucs-d4b573e5.so.0.0.0 libucs.so.0 \
    --set-rpath "$overlay_prefix/lib:$original_nixl_libs" \
    "$stage_prefix/nixl/plugins/libplugin_UCX.so"

readonly observed_version="$(
    LD_LIBRARY_PATH="$stage_prefix/lib" "$stage_prefix/bin/ucx_info" -v |
        sed -n 's/^# Library version: \([^ ]*\).*$/\1/p' | head -1
)"
[[ "$observed_version" == "$UCX_VERSION" ]] || {
    echo "expected UCX $UCX_VERSION, found $observed_version" >&2
    exit 1
}

{
    echo "ucx_commit=$UCX_COMMIT"
    echo "ucx_version=$UCX_VERSION"
    echo "ucx_archive_sha256=$UCX_ARCHIVE_SHA256"
    echo "gpunetio_commit=$GPUNETIO_COMMIT"
    echo "gpunetio_archive_sha256=$GPUNETIO_ARCHIVE_SHA256"
    echo "patchelf_version=$PATCHELF_VERSION"
    echo "patchelf_archive_sha256=$PATCHELF_ARCHIVE_SHA256"
    echo "patchelf_sha256=$PATCHELF_SHA256"
    echo "nixl_version=1.3.1"
    echo "nixl_plugin_sha256=$NIXL_PLUGIN_SHA256"
    echo "cuda_version=$(nvcc --version | sed -n 's/.*release \([^,]*\).*/\1/p' | tail -1)"
    echo "glibc_version=$(ldd --version | head -1)"
    echo "build_packages:"
    dpkg-query -W -f='${Package}=${Version}\n' \
        autoconf automake cuda-nvml-dev-13-0 flex gcc g++ libibumad-dev \
        libibverbs-dev libnl-3-dev libnl-route-3-dev libnuma-dev libtool \
        make pkg-config
    echo "plugin_needed:"
    "$patchelf_bin" --print-needed "$stage_prefix/nixl/plugins/libplugin_UCX.so"
    echo "plugin_rpath=$(
        "$patchelf_bin" --print-rpath "$stage_prefix/nixl/plugins/libplugin_UCX.so"
    )"
} >"$stage_prefix/inferencex-ucx-overlay.manifest"

(
    cd "$stage_prefix"
    sha256sum \
        lib/libucp.so.0.0.0 \
        lib/libucs.so.0.0.0 \
        lib/libuct.so.0.0.0 \
        lib/ucx/libuct_cuda.so.0.0.0 \
        nixl/plugins/libplugin_UCX.so \
        >inferencex-ucx-overlay.sha256
)

[[ ! -e "$overlay_prefix" ]] || {
    echo "overlay target appeared while building: $overlay_prefix" >&2
    exit 1
}
mv "$stage_prefix" "$overlay_prefix"
validate_existing
echo "built and validated UCX overlay: $overlay_prefix"
