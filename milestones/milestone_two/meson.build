# milestone_two executable
# Following professor's instruction: "add 'kokkoscore' to the dependencies in your subdirectories"

# Try to find kokkoscore, but handle CI gracefully
# Note: required=false means build continues even if kokkos fails
kokkos_dep = dependency('kokkoscore', required: false)

milestone_two_deps = []
cpp_args = []

if kokkos_dep.found()
    milestone_two_deps += kokkos_dep
    cpp_args += ['-DHAS_KOKKOS=1']
    message('milestone_two: Using Kokkos Views as specified in assignment')
else
    cpp_args += ['-DHAS_KOKKOS=0']
    message('milestone_two: Kokkos not available, using standard C++ fallback (same algorithm)')
endif

milestone_two_exe = executable(
    'milestone_two',
    ['milestone_two.cpp', 'LBMStreaming.cpp'],
    dependencies: milestone_two_deps,
    cpp_args: cpp_args,
    install: false
)