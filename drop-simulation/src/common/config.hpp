#pragma once

#include <string>

// Configuracion compartida por todas las versiones GPU. Los valores son
// identicos a los de src/main.cpp (version CPU) para que la validacion
// CPU vs GPU compare exactamente la misma simulacion.
struct Config {
    int width = 640;
    int height = 640;
    double seconds = 30.0;
    int fps = 30;
    int steps_per_frame = 1;
    float wave_speed = 0.45f;
    float damping = 0.006f;
    float edge_damping = 0.035f;
    float drop_radius = 18.0f;
    float drop_strength = 1.0f;
    std::string output = "output/drop_simulation.mp4";
};
