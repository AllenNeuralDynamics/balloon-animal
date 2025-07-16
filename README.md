# balloon-animal

Central repository containing code to run every step of the balloon-animal track generation pipeline.

The NPY contains timeseries of the volumetric and marble gaussians. Gaussians are parametrized by location/scale/rotation and RGBA.
NPY file format:
- Buffers associated with base volumes:
    - 'means3D': (N, B, 3)
    - 'rgb_colors': (N, B, 3)
    - 'logit_opacities': (N, B, 1)
    - 'log_scales': (N, B, 1)
    - 'unnorm_rotations': (N, B, 4)
Use means3D for center tracks.
The buffers 'rgb_colors', 'logit_opacities', 'log_scales', and 'unnorm_rotations' are frozen in tracking.

- Buffers associated with surface marbles:
    - 'surface_means3D': (N, T, 3)
    - 'surface_rgb':(N, T, 3)
    - 'surface_logit_opacities': (N, T, 1)
    - 'surface_log_scale': (N, T, 1)
    - 'surface_unnorm_rotations': (N, T, 4)
Use surface_means3D for surface tracks.
The buffers 'surface_rgb', 'surface_logit_opacities', 'surface_log_scale', and 'surface_unnorm_rotations' are frozen in tracking.

NOTE:
To keep this repository clean, heavy dependencies like Pytorch are keep separate. Please install this natively.
