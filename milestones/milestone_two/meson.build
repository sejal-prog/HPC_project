kokkos_dep = dependency('kokkoscore', required: false, fallback: ['kokkos', 'kokkoscore_dep'])

milestone_two_deps = []
if kokkos_dep.found()
    milestone_two_deps += kokkos_dep
endif

milestone_two_exe = executable(
    'milestone_two',
    ['milestone_two.cpp', 'LBMStreaming.cpp'],
    dependencies: milestone_two_deps,
    install: false
)