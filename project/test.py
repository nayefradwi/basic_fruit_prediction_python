import unittest

class TestingHomeworkFunctions(unittest.TestCase):
        def test_training_one_epoch(self):
            # self.assertEqual(1, result)
            pass

        def test_prediction_after_training_one_epoch(self):
            pass

        def testing_one_perceptron(self):
            pass

        def test_voting(self):
            pass

        

# uncomment if you want to run unit tests
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestingHomeworkFunctions)
    unittest.TextTestRunner(verbosity=2).run(suite)
