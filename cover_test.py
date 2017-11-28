import unittest


def count_white_spaces(str):

	count = 0

	for i in range (0, len(str)):

		if (str[i] == ' '):
			count = count + 1

	return (count)

class Test_Count(unittest.TestCase):

	
		
	testStr1 = " "
	testStr2 = "h"
	testStr3 = ""

	def test1(self):

		self.assertEqual(count_white_spaces(self.testStr1), 1)
		self.assertEqual(count_white_spaces(self.testStr2), 0)
		self.assertEqual(count_white_spaces(self.testStr3), 0)
		
		
if __name__ == '__main__':
	unittest.main()

