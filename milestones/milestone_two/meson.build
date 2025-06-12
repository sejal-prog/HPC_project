# milestone_two executable
# Add kokkoscore dependency at subdirectory level as instructed
kokkoscore = dependency('kokkoscore', fallback: ['kokkos', 'kokkoscore_dep'])

milestone_two_exe = executable(
    'milestone_two',
    ['milestone_two.cpp', 'LBMStreaming.cpp'],
    dependencies: [kokkoscore],
    install: false
)