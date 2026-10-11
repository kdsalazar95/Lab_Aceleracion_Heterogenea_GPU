# RGBA Image Color Inversion

CPU and CUDA implementations in C++ and OpenCV for RGBA color inversion.

The program reads an image with OpenCV, converts it to 8-bit BGRA when needed,
inverts the color channels, preserves alpha, and writes the result.

## Build

```bash
meson setup build
meson compile -C build
```

## Run

```bash
./build/invert_cpu input.png output.png
./build/invert_cuda input.png output_cuda.png
./build/invert_camera
./build/invert_camera 1
./build/invert_camera_ring
./build/invert_camera_ring 1
./build/invert_camera_ring_cpu
./build/invert_camera_ring_cpu 1
```

OpenCV stores RGBA images as BGRA in memory, so the CPU loop inverts channels
B, G, and R while leaving A unchanged. The CUDA versions use the same BGRA
layout, explicit host/device copies, and one `uchar4` pixel per GPU thread. The
camera version captures live frames from OpenCV, converts them to BGRA, inverts
them with CUDA, and shows the result until `q` or Esc is pressed.

`invert_camera_ring` uses three threads and bounded ring queues: one thread
captures frames, one thread executes CUDA with persistent device buffers, and
one thread displays the processed frames. This overlaps capture, GPU work, and
display to hide latency.

`invert_camera_ring_cpu` uses the same three-thread ring pipeline, but the
processing stage performs the inversion on the CPU.
