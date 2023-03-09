import unittest
import sys

if __name__ == "__main__":

    test_dirs = ["self_ck_attn"]

    runner = unittest.TextTestRunner(verbosity=2)

    errcode = 0

    for test_dir in test_dirs:
        suite = unittest.TestLoader().discover(test_dir)

        print("\nExecuting tests from " + test_dir)

        result = runner.run(suite)

        if not result.wasSuccessful():
            errcode = 1

    sys.exit(errcode)
