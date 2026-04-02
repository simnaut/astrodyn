# Build Trick + JEOD on Rocky Linux 9 for generating reference trajectories.
#
# Usage (from bevy_jeod project root, with trick and jeod as siblings):
#   docker build -f trick/Dockerfile -t jeod-trick ..
#   docker run --rm \
#     -v $(pwd)/test_data:/output \
#     -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro \
#     jeod-trick
#
# The build context is the parent directory (..) so that both trick/ and jeod/
# are accessible via COPY. BEVY_JEOD_DIR defaults to "bevy_jeod" but can be
# overridden if the repo was cloned under a different name:
#   docker build -f trick/Dockerfile --build-arg BEVY_JEOD_DIR=my_clone -t jeod-trick ..

FROM rockylinux/rockylinux:9

ARG BEVY_JEOD_DIR=bevy_jeod

# ── System dependencies (from Trick's test_linux.yml Rocky 9 config) ──
RUN dnf -y install epel-release && \
    dnf -y update && \
    dnf install -y 'dnf-command(config-manager)' && \
    dnf config-manager --enable crb && \
    dnf install -y \
        bison clang clang-devel cmake diffutils flex gcc gcc-c++ git \
        java-21-openjdk-devel libxml2-devel llvm llvm-devel llvm-static \
        make maven ncurses-devel openmotif openmotif-devel perl \
        perl-Digest-MD5 python3-devel swig udunits2 udunits2-devel \
        which zlib-devel zip gdb gtest-devel gmock-devel && \
    dnf clean all

# ── Build Trick ──
COPY trick /trick
WORKDIR /trick
RUN ./configure && \
    make -j$(nproc)

ENV TRICK_HOME=/trick
ENV PATH="${TRICK_HOME}/bin:${PATH}"

# ── Copy JEOD ──
COPY jeod /jeod
ENV JEOD_HOME=/jeod

# Build the JEOD static library (clean any host build artifacts first)
WORKDIR /jeod
RUN rm -rf build build_unit_test lib_jeod_unit_test && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DTRICK_HOME=/trick && \
    make -j$(nproc)

# ── Output volume ──
VOLUME /output

# The generate_references.sh script is mounted at runtime, not baked into the
# image.  This avoids Docker layer cache issues when only the script changes:
#   docker run --rm -e FORCE=1 \
#     -v $(pwd)/test_data:/output \
#     -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro \
#     jeod-trick
CMD ["/generate_references.sh"]
