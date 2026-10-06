import contextlib
import io
import unittest
from robotic_arm.controllers.pick_place_controller import run


class PickPlaceTests(unittest.TestCase):
    def test_physical_grasp_lift_and_release(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(render=False)
        self.assertTrue(result["grasped_with_both_fingers"])
        self.assertTrue(result["lifted"])
        self.assertTrue(result["released"])
        self.assertTrue(result["resting"])
        self.assertTrue(result["success"])
        self.assertLess(result["placement_error_m"], 0.05)

    def test_slippery_light_cube(self):
        result=run(render=False,source=(0.418, -0.116),destination=(0.447,0.144),mass=0.04,friction=0.5,verbose=False)
        self.assertTrue(result["success"])
        self.assertFalse(result["ground_contact"])

    def test_invalid_scene_parameters(self):
        for parameters in [{"mass":0},{"mass":float("nan")},{"friction":-1},{"source":[0,0,0]},
                           {"destination":[float("inf"),0]}]:
            with self.assertRaises(ValueError):
                run(render=False,**parameters)
