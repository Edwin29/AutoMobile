import unittest
from probe import validate,duty_at,safety_reason
from collect import summarize


class Tests(unittest.TestCase):
    def sample(self,**kw):
        return dict(voltage=5.05,voltage_at=10,temp=45,throttled=0)|kw

    def test_ramp_and_absolute_end(self):
        self.assertEqual(duty_at(0,.2,1,.5),0)
        self.assertAlmostEqual(duty_at(.5,.2,1,.5),.1)
        self.assertEqual(duty_at(1.2,.2,1,.5),.2)
        self.assertAlmostEqual(duty_at(2,.2,1,.5),.1)
        self.assertEqual(duty_at(2.5,.2,1,.5),0)

    def test_parameter_bounds(self):
        for duty in [-.1,.6,float('nan')]:
            with self.assertRaises(ValueError):validate('left','floor',duty,1,.5,1000)

    def test_heartbeat_loss(self):
        self.assertEqual(safety_reason(self.sample(),10,8,False),'heartbeat_timeout')

    def test_voltage_floor(self):
        self.assertEqual(safety_reason(self.sample(voltage=4.79),10,10,False),'input_voltage_below_4.80V')

    def test_stale_sensor(self):
        self.assertEqual(safety_reason(self.sample(voltage_at=9),10,10,False),'voltage_telemetry_stale')

    def test_failed_sensor(self):
        self.assertEqual(safety_reason(self.sample(voltage=None),10,10,False),'voltage_read_failed')

    def test_temperature(self):
        self.assertEqual(safety_reason(self.sample(temp=81),10,10,False),'temperature_at_least_80C')

    def test_current_undervoltage_not_old_history(self):
        self.assertIsNone(safety_reason(self.sample(throttled=0x10000),10,10,False))
        self.assertEqual(safety_reason(self.sample(throttled=1),10,10,False),'current_undervoltage_flag')

    def test_operator_stop(self):
        self.assertEqual(safety_reason(self.sample(),10,10,True),'operator_or_connection_stop')

    def test_disconnect_is_not_power_proof(self):
        self.assertIn('unconfirmed',summarize([])['end_reason'])


if __name__=='__main__':unittest.main()
