"""
Code for stages of segmentation pipeline.
"""

from base import OmDetInput, SAM2Input

def run_omdet(
    src: OmDetInput
) -> SAM2Input:
    pass

def run_sam2(
    src: SAM2Input
) -> SchedulerInput:
    pass
