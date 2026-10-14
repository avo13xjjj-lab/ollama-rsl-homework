import unittest
from rsl import errors

class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.expected = dict(person='李四', date=None, location='杭州', event='参加会议', budget=1500)
    def test_valid(self):
        self.assertEqual(errors(self.expected, self.expected), [])
    def test_boolean_is_not_amount(self):
        self.assertTrue(errors({**self.expected, 'budget':True}, {**self.expected,'budget':1}))
    def test_string_amount_is_rejected(self):
        self.assertTrue(errors({**self.expected,'budget':'1500'},self.expected))
    def test_missing_null_key_is_rejected(self):
        value=dict(self.expected); del value['date']
        self.assertTrue(errors(value,self.expected))
    def test_invented_date_is_rejected(self):
        self.assertTrue(errors({**self.expected,'date':'2026-11-01'},self.expected))
    def test_array_is_rejected(self):
        self.assertTrue(errors([],self.expected))

if __name__=='__main__': unittest.main()
