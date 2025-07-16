"""
Visualize gaussian/point tracks in Rerun.
"""

import numpy as np
import rerun as rr
import torch

class Skeleton:
    def __init__(self, points=None, edges=None):
        self.points = points
        self.edges = edges

class RerunSession:
    def __init__(self, ply_files: list[str], skeleton_tracks: list[list[Skeleton]] | None):
        """
        Args:
            ply_files (list[str]): Pass in list of ply files to visualize
            skeleton_tracks (list[list[Skeleton]]): Optional list of SkeletonTracks
        """
        assert len(ply_files) == len(skeleton_tracks), \
            print("Number of ply files much match number fo skeleton tracks")

        first_point_track = np.load(ply_files[0])
        first_skeleton_track = skeleton_tracks[0]
        assert len(first_point_track['means3D']) == len(first_skeleton_track)

        self.ply_files = ply_files
        self.skeleton_tracks = skeleton_tracks

    def visualize(self):
        rr.init('Track Visualization', spawn=True)
        for f, ply in enumerate(self.ply_files):
            print(f'Plotting {ply}')
            gaussian_timeseries = np.load(ply)

            base_means = torch.Tensor(gaussian_timeseries['means3D'])
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

            # Plot skeleton if present
            if self.skeleton_tracks:
                for i, s in enumerate(self.skeleton_tracks[f]):
                    rr.set_time_seconds("iteration", (f * len(gaussian_timeseries)) + i)
                    rr.log(
                        "pose",
                        rr.Points3D(
                            s.points,
                            keypoint_ids = range(len(s.points))
                        )
                    )

                    connections = []
                    for start_id, end_id in s.edges:
                        connections.append([s.points[start_id], s.points[end_id]])
                    rr.log(
                        "pose/connections",
                        rr.LineStrips3D(
                            strips=connections,
                            colors=np.array([1.0, 0.0, 0.0]),
                            radii=0.5
                        )
                    )


if __name__ == '__main__':
    # Example Usage:
    ply_files = [
        "tracks_0_20.npz", "tracks_20_40.npz"
    ]

    pose_info = ...
    skeleton_indices = [
        (0, 1),    # HeadF to HeadB
        (0, 2),    # HeadF to HeadL
        (1, 3),    # HeadB to SpineF
        (2, 3),    # HeadL to SpineF
        (3, 4),    # SpineF to SpineM
        (4, 5),    # SpineM to SpineL
        (5, 8),    # SpineL to HipL
        (5, 9),    # SpineL to HipR
        (12, 10),  # ShoulderL to ElbowL
        (10, 11),  # ElbowL to ArmL
        (13, 14),  # ShoulderR to ElbowR
        (14, 15),  # ElbowR to ArmR
        (3, 12),   # SpineF to ShoulderL
        (3, 13),   # SpineF to ShoulderR
        (8, 17),   # HipL to KneeL
        (17, 18),  # KneeL to ShinL
        (9, 16),   # HipR to KneeR
        (16, 19)   # KneeR to ShinR
    ]
    skeleton_tracks = []
    num_frames = 40
    track_interval = 20
    for i in range(0, num_frames, track_interval):
        track = []
        interval = pose_info[i:i+track_interval]
        for j in range(track_interval):
            points = interval[j]
            track.append(Skeleton(points=points, edges=skeleton_indices))
        skeleton_tracks.append(track)

    rr_session = RerunSession(ply_files, skeleton_tracks)
    rr_session.visualize()