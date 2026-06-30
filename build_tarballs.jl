# Note that this script can accept some limited command-line arguments, run
# `julia build_tarballs.jl --help` to see a usage message.
using BinaryBuilder, Pkg
using Base.BinaryPlatforms
const YGGDRASIL_DIR = "./yggdrasil-files/"
include(joinpath(YGGDRASIL_DIR, "platforms", "macos_sdks.jl"))
include(joinpath(YGGDRASIL_DIR, "platforms", "mpi.jl"))

name = "libdealii_trixi_paper2026"
version = v"0.1.4"

# collection of sources required to complete build
sources = [
  DirectorySource("./dealii-trixi/")
]

# bash recipe for building across all platforms
script = raw"""
# build writes to /tmp, which is a small tmpfs in our sandbox
# make it use the workspace instead
export TMPDIR=${WORKSPACE}/tmpdir
mkdir ${TMPDIR}

cmake_options=(
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$prefix \
  -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TARGET_TOOLCHAIN} \
)

if [[ ${bb_full_target} == *mpiabi* ]]; then
  # MPIABI splits the C and Fortran MPI bindings
  cmake_options+=(
    -DMPI_C_COMPILER=mpicc
    -DMPI_Fortran_COMPILER=mpifc
  )
else
  cmake_options+=(
    -DMPI_HOME=${prefix}
  )
fi

export MPITRAMPOLINE_CC="${CC}"
export MPITRAMPOLINE_CXX="${CXX}"
export MPITRAMPOLINE_FC="${FC}"

# build libdealii-trixi
mkdir build
cd build
cmake "${cmake_options[@]}"  ..
make -j${nproc}

# the licence of the package is expected to be in this directory
mkdir -p ${prefix}/share/licenses/libdealii-trixi
install_license ../LICENSE.md

make install
"""

augment_platform_block = """
  using Base.BinaryPlatforms
  $(MPI.augment)
  augment_platform!(platform::Platform) = augment_mpi!(platform)
"""

# these are the platforms we will build for by default, unless further platforms
# are passed in on the command line
platforms = supported_platforms()

platforms, platform_dependencies = MPI.augment_platforms(platforms)

# due to deal.II
## due to P4est
platforms = filter(p -> !(Sys.iswindows(p) && nbits(p) == 32), platforms)
platforms = filter(p -> !(p["mpi"] in ["openmpi", "mpiabi"]), platforms)
## due to Kokkos
platforms = filter(p -> nbits(p) == 64, platforms)
## due to HDF5
sources, script = require_macos_sdk("14.0", sources, script)
## due to GSL
platforms = filter(p -> p["arch"] != "riscv64", platforms)
platforms = filter(p -> p["os"] != "freebsd", platforms)
## Windows builds with MinGW are not supported by deal.II
## see https://github.com/dealii/dealii/wiki/Windows#cygwin--minggw
platforms = filter(p -> !Sys.iswindows(p), platforms)
## powerpc64le builds fail due to missing support for long doubles in Boost
## furthermore: https://github.com/dealii/dealii/wiki/Power-architecture
platforms = filter(p -> p["arch"] != "powerpc64le", platforms)


platforms = expand_cxxstring_abis(platforms)

# the products that we will ensure are always built
products = [
    LibraryProduct("libdealii_trixi_paper2026", :libdealii_trixi_paper2026),
]

# dependencies that must be installed before this package can be built
# commented dependencies might be added in the future but stay here for reference
dependencies = [
  RuntimeDependency(PackageSpec(name="MPIPreferences",
                                uuid="3da0fdf6-3ccc-4f1b-acd9-58baa6c99267")),
  Dependency(PackageSpec(name="deal_II_jll",
                         uuid="5bf063c2-051f-51d9-b770-4d3ca06472e2")),
]
append!(dependencies, platform_dependencies)

# don't look for `mpiwrapper.so` when BinaryBuilder examines and `dlopen`s the
# shared libraries. (MPItrampoline will skip its automatic initialization.)
ENV["MPITRAMPOLINE_DELAY_INIT"] = "1"

build_tarballs(ARGS, name, version, sources, script, platforms, products,
               dependencies; augment_platform_block, julia_compat="1.10",
               dont_dlopen=true, preferred_gcc_version = v"9.1.0")
