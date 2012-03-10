import unittest
import fieldparsers

class FieldParsersTests(unittest.TestCase):

    #def test_partition(self):
    #    x = [1, 2, 2, 1]
    #    y = [5, 6, 7, 8]
    #    (result, errors) = fieldparsers.partition_values(x, y, [1, 2]);
    #    self.assertEqual({1: [5, 8], 2:[6, 7]}, result)

    def test_parse_employment_years(self):
        self.assertEquals(1, fieldparsers.parse_employment_years('1 year'))
        self.assertEquals(0, fieldparsers.parse_employment_years('<1 year'))
        self.assertEquals(10, fieldparsers.parse_employment_years('10+ years'))

if __name__ == '__main__':
    unittest.main()
