import unittest

class BaseUnitTest(unittest.TestCase):

  def eq(self, expected, target):
    return self.assertEqual(expected, target)

  def almosteq(self, expected, target, tolerance):
    if isinstance(expected, list):
      curry = lambda zipped: self.almosteq(zipped[0], zipped[1], tolerance)
      map(curry, zip(expected, target))
    else:
      match = expected - tolerance <= target <= expected + tolerance
      if not match:
        return self.fail("%s and %s do not match with tolerance=%f" % (expected, target, tolerance))

  def neq(self, expected, target):
    return self.assertNotEqual(expected, target)

  def true(self, target):
    return self.assertTrue(target)

  def false(self, target):
    return self.assertFalse(target)

  def include(self, target, source):
    return self.assertIn(target, source)

  def not_include(self, target, source):
    return self.assertNotIn(target, source)

  def debug(self):
    from nose.tools import set_trace; set_trace()

