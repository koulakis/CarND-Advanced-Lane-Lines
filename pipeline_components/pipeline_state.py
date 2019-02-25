from typing import List, Dict, Any


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
            step_number: int,
            errors: List[Dict[str, Any]],
            left_fit=None,
            right_fit=None):

        self.step_number = step_number
        self.left_fit = left_fit
        self.right_fit = right_fit
        self.errors = errors

    def add_error(self, error):
        self.errors.append(error)

    def set_left_fit(self, left_fit):
        self.left_fit = left_fit

    def set_right_fit(self, right_fit):
        self.right_fit = right_fit

    def __str__(self):
        return str({
            'sterp_number': self.step_number,
            'left_fit': self.left_fit,
            'right_fit': self.right_fit,
            'errors': self.errors})


class TransformContext:
    def __init__(self, component_name: str, state: Dict[str, Any]):
        self.state = state
        self.component_name = component_name

    def __enter__(self):
        return self.state

    def __exit__(self, exc_type, exc_val, exc_tb):
        if (exc_val is not None) and (len(self.state['steps'][-1].errors) == 0):
            if exc_type is not TransformError:
                return False

            self.state['steps'][-1].errors.append(
                {
                    'component': self.component_name,
                    'exception': exc_val.error_details()})
            return True
        return True
