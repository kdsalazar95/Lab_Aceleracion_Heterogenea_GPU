project(
  'image-color-inversion',
  ['cpp', 'cuda'],
  version: '0.1.0',
  default_options: [
    'cpp_std=c++17',
    'warning_level=3',
    'buildtype=release',
  ],
)

opencv = dependency('opencv4', modules: ['core', 'highgui', 'imgcodecs', 'imgproc', 'videoio'])

executable(
  'invert_cpu',
  'src/main.cpp',
  dependencies: opencv,
  install: false,
)

executable(
  'invert_cuda',
  [
    'src/main-cuda.cpp',
    'src/invert_cuda.cu',
  ],
  cuda_args: ['-Xcompiler=-Wno-pedantic'],
  dependencies: opencv,
  install: false,
)

executable(
  'invert_camera',
  [
    'src/main-camera.cpp',
    'src/invert_cuda.cu',
  ],
  cuda_args: ['-Xcompiler=-Wno-pedantic'],
  dependencies: opencv,
  install: false,
)

executable(
  'invert_camera_ring',
  [
    'src/main-camera-ring.cpp',
    'src/camera_ring_cuda.cu',
  ],
  cuda_args: ['-Xcompiler=-Wno-pedantic'],
  dependencies: opencv,
  install: false,
)

executable(
  'invert_camera_ring_cpu',
  'src/main-camera-ring-cpu.cpp',
  dependencies: opencv,
  install: false,
)
