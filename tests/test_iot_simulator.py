import unittest

from iot_simulator import VirtualIoTController


class TestVirtualIoTController(unittest.TestCase):

    def setUp(self):
        self.device = VirtualIoTController()

    def test_initial_state_is_locked(self):
        state = self.device.get_status()

        self.assertEqual(state["door_status"], "LOCKED")
        self.assertEqual(state["servo_angle"], 0)
        self.assertTrue(state["online"])

    def test_accept_unlocks_door(self):
        state = self.device.apply_authentication_result(
            "ACCEPT", "A01", "A01", 3, 32, 5.2272, 0.8776
        )

        self.assertEqual(state["door_status"], "UNLOCKED")
        self.assertEqual(state["servo_angle"], 90)
        self.assertEqual(state["last_command"], "UNLOCK")

    def test_reject_keeps_door_locked(self):
        state = self.device.apply_authentication_result(
            "REJECT", "A01", "A02", 3, 32, -40.3397, 0.8776
        )

        self.assertEqual(state["door_status"], "LOCKED")
        self.assertEqual(state["servo_angle"], 0)
        self.assertEqual(state["last_command"], "LOCK")

    def test_rejection_relocks_after_acceptance(self):
        self.device.apply_authentication_result(
            "ACCEPT", "A01", "A01", 3, 32, 5.2272, 0.8776
        )

        state = self.device.apply_authentication_result(
            "REJECT", "A01", "A02", 3, 32, -40.3397, 0.8776
        )

        self.assertEqual(state["door_status"], "LOCKED")
        self.assertEqual(state["servo_angle"], 0)

    def test_invalid_decision_is_rejected(self):
        with self.assertRaises(ValueError):
            self.device.apply_authentication_result(
                "MAYBE", "A01", "A01", 3, 32, 1.0, 0.8
            )

    def test_event_history_is_recorded(self):
        state = self.device.apply_authentication_result(
            "ACCEPT", "A01", "A01", 3, 32, 5.2272, 0.8776
        )

        self.assertEqual(len(state["history"]), 1)
        self.assertEqual(
            state["history"][0]["authentication_decision"], "ACCEPT"
        )


if __name__ == "__main__":
    unittest.main()
