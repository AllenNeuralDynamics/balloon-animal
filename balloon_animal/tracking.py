"""
Code for stages of tracking pipeline.
"""

from balloon_animal.base import (SchedulerInput,
                  TrackBalloonInput,
                  BalloonSequence)

def schedule_tracking(
    src: SchedulerInput
) -> TrackBalloonInput:
    pass

def track_balloons(
    src: TrackBalloonInput
) -> BalloonSequence:
    pass

def combine_balloon_tracks(
    seq: BalloonSequence
) -> BalloonSequence:
    pass