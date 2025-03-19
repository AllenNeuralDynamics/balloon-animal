"""
Standard base objects that standardize 
what contents should exist between pipeline stages. 

Base objects do not hold data in RAM.
They only hold filesystem paths
and serve as validation classes for self-documentation. 
"""

# Add something like this to each class, likely unique to each stage.
def validate_image_directory():
    pass
def validate_yml_file():  
    pass

class OmDetInput:
    def __init__(self):
        pass

class SAM2Input:
    def __init__(self):
        pass

class SchedulerInput:
    def __init__(self):
        pass

class TrackBalloonInput:
    def __init__(self):
        pass

class BalloonSequence:
    def __init__(self):
        pass