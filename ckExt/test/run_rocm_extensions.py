import unittest
import sys

if __name__ == "__main__":

    #test_dirs = ["groupbn", "layer_norm", "multihead_attn", "transducer", "focal_loss", "index_mul_2d", "ck_attn", "."] # "." for test_label_smoothing.py
    #test_dirs = ["ck_attn", "."] # "." for test_label_smoothing.py
    test_dirs = ["self_ck_attn"] # "." for test_label_smoothing.py
    ROCM_BLACKLIST = [
    ]

    runner = unittest.TextTestRunner(verbosity=2)

    errcode = 0

    for test_dir in test_dirs:
        if test_dir in ROCM_BLACKLIST:
            continue
        suite = unittest.TestLoader().discover(test_dir)

        print("\nExecuting tests from " + test_dir)

        result = runner.run(suite)

        if not result.wasSuccessful():
            errcode = 1

    sys.exit(errcode)
