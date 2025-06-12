# milestone_two executable with Kokkos
kokkos_dep = dependency('kokkoscore', required: false, fallback: ['kokkos', 'kokkoscore_dep'])

milestone_two_deps = []
if kokkos_dep.found()
    milestone_two_deps += kokkos_dep
    message('Building milestone_two with Kokkos support')
else
    error('Kokkos is required for milestone_two as specified in the assignment')
endif

milestone_two_exe = executable(
    'milestone_two',
    ['milestone_two.cpp', 'LBMStreaming.cpp'],
    dependencies: milestone_two_deps,
    install: false
)