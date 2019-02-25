from typing import List, Dict, Any
import numpy as np


class TransformError(Exception):
    def __init__(self, message, captured_exception=None):
        super().__init__("'message': {}, 'captured_exception': {}".format(
            message,
            'None' if captured_exception is None else captured_exception))

        self.message = message
        self.captured_exception = captured_exception

    def error_details(self):
        return {'message': self.message, 'captured_exception': self.captured_exception}


class SingleStepState:
    def __init__(
            self,
            data=None,
            errors: List[TransformError]=[],
            extracted_lane_information: Dict[str, Any]={
                'left_fit': None,
                'right_fit': None
            },
            image: np.dtype('uint8')=None):

        self.data = data
        self.errors = errors
        self.extracted_lane_information = extracted_lane_information
        self.image = image


class TransformContext:
    def __init__(self, component_name: str, state: List[SingleStepState]):
        self.state = state
        self.component_name = component_name

    def __enter__(self):
        return self.state

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type != TransformError:
            return False

        if exc_val is not None:
            self.state[-1].errors.append(exc_val.error_details())
            return True
