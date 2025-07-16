"""
Visualize gaussian/point tracks in Rerun.
"""

import numpy as np
import rerun as rr
import torch

class RerunSession:
    def __init__(self, ply_files: list[str]):
        """
        Args:
            ply_files (list[str]): Pass in list of ply files to visualize
        """
        self.ply_files = ply_files

    def visualize(self):
        rr.init('Track Visualization', spawn=True)
        for f, ply in enumerate(self.ply_files):
            print(f'Plotting {ply}')
            gaussian_timeseries = np.load(ply)

            base_means = torch.Tensor(gaussian_timeseries['means3D'])
            # base_semi_axes = torch.Tensor(np.exp(gaussian_timeseries['log_scales'])* 2)
            # base_quaternions = torch.Tensor(gaussian_timeseries['unnorm_rotations'])

            surface_means = torch.Tensor(gaussian_timeseries['surface_means3D'])
            surface_rgb = torch.Tensor(np.clip(gaussian_timeseries['surface_rgb'], 0, 1))

            for i in range(len(gaussian_timeseries)):
                print(f'iteration: {(f * len(gaussian_timeseries)) + i}')
                rr.set_time_seconds("iteration", (f * len(gaussian_timeseries)) + i)

                # Log Base Gaussian Center
                curr_centers = base_means[i, ...]
                rr.log(
                    "centers",
                    rr.Points3D(
                        positions=curr_centers,
                        colors=np.array([0.0, 1.0, 1.0]),
                        radii=np.full(len(curr_centers), 1.0)
                    )
                )

                # Log Base Gaussian Surface Pts
                curr_surface_pts = surface_means[i, ...]
                curr_surface_color = surface_rgb[i, ...]
                rr.log(
                    f"sampled_surface",
                    rr.Points3D(
                        positions=curr_surface_pts,
                        colors=curr_surface_color,
                        radii=np.full(len(curr_surface_pts), 1.0)
                    ),
                )

            num_surface_pts = len(surface_means[0, ...])
            track_colors = np.random.rand(num_surface_pts, 3)
            track_colors = np.clip(track_colors + 0.3, 0, 1)
            num_centers = len(base_means[0, ...])
            center_colors = np.random.rand(num_centers, 3)
            center_colors = np.clip(center_colors + 0.3, 0, 1)

            # Plot center tracks
            for center_idx in range(num_centers):
                track_positions = base_means[:, center_idx, :]
                # Log this track as a single entity that spans all time steps
                for time_idx in range(len(gaussian_timeseries)):
                    rr.set_time_seconds("iteration", (f * len(gaussian_timeseries)) + time_idx)

                    # Get positions up to the current time step
                    current_track_segment = track_positions[:time_idx+1]

                    rr.log(
                        f"center_tracks/track_{center_idx}",
                        rr.LineStrips3D(
                            strips=[current_track_segment],
                            colors=center_colors[center_idx],
                            radii=0.5  # Adjust line thickness as needed
                        )
                    )

            # Plot sample of surface tracks
            visible_tracks = range(0, num_surface_pts, num_surface_pts // 200)
            for track_idx in visible_tracks:
                # Extract positions for this track across all time steps
                track_positions = surface_means[:, track_idx, :]

                # Log this track as a single entity that spans all time steps
                for time_idx in range(len(gaussian_timeseries)):
                    rr.set_time_seconds("iteration", (f * len(gaussian_timeseries)) + time_idx)

                    # Get positions up to the current time step
                    current_track_segment = track_positions[:time_idx+1]

                    rr.log(
                        f"tracks/track_{track_idx}",
                        rr.LineStrips3D(
                            strips=[current_track_segment],
                            colors=track_colors[track_idx],
                            radii=0.5  # Adjust line thickness as needed
                        )
                    )

if __name__ == '__main__':
    # Example Usage:
    ply_files = [
        "tracks_0_20.npz", "tracks_20_40.npz"
    ]
    rr_session = RerunSession(ply_files)
    rr_session.visualize()