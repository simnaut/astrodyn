# Build Trick + JEOD on Rocky Linux 9 for generating reference trajectories.
#
# Usage:
#   docker build -t jeod-trick trick/
#   docker run --rm -v $(pwd)/test_data:/output jeod-trick
#
# This produces CSV reference trajectories in test_data/ that bevy_jeod
# integration tests compare against (Tier 3 cross-validation).

FROM rockylinux/rockylinux:9

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
    make -j$(nproc) && \
    make install

ENV TRICK_HOME=/trick
ENV PATH="${TRICK_HOME}/bin:${PATH}"

# ── Build JEOD ──
COPY jeod /jeod
ENV JEOD_HOME=/jeod

# Build the JEOD static library (used by verification sims)
WORKDIR /jeod
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# ── Copy the reference data generation script ──
COPY trick/generate_references.sh /generate_references.sh
RUN chmod +x /generate_references.sh

# ── Output volume ──
VOLUME /output

CMD ["/generate_references.sh"]
