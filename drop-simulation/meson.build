project(
  'drop-simulation',
  'cpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++17',
    'warning_level=3',
    'buildtype=release',
  ],
)

opencv = dependency('opencv4', modules: ['core', 'imgproc', 'imgcodecs', 'videoio'])

executable(
  'drop_simulation',
  'src/main.cpp',
  dependencies: opencv,
  install: false,
)

# Versiones GPU: solo se compilan si hay nvcc (Jetson Nano). En una maquina
# sin CUDA se omiten y la version CPU sigue compilando normalmente.
if add_languages('cuda', required: false)
  # Jetson Nano = Maxwell, compute capability 5.3 (sm_53).
  cuda_arch = get_option('cuda_arch')
  gpu_cuda_args = [
    '-gencode=arch=compute_@0@,code=sm_@0@'.format(cuda_arch),
    '-Xcompiler=-Wno-pedantic',
  ]

  executable(
    'drop_simulation_gpu_v0',
    [
      'src/gpu/v0_baseline/main.cpp',
      'src/gpu/v0_baseline/kernels.cu',
    ],
    cuda_args: gpu_cuda_args,
    dependencies: opencv,
    install: false,
  )
else
  message('nvcc no encontrado: se omiten los targets GPU')
endif
