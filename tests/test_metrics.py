import keras
import keras.backend
from .. import keras_metrics
import itertools
import numpy
import unittest
import os.path


def ref_true_pos(y_true, y_pred):
    return numpy.sum(numpy.logical_and(numpy.round(y_pred) == 1, y_true == 1))


def ref_false_pos(y_true, y_pred):
    return numpy.sum(numpy.logical_and(numpy.round(y_pred) == 1, y_true == 0))


def ref_true_neg(y_true, y_pred):
    return numpy.sum(numpy.logical_and(numpy.round(y_pred) == 0, y_true == 0))


def ref_false_neg(y_true, y_pred):
    return numpy.sum(numpy.logical_and(numpy.round(y_pred) == 0, y_true == 1))

class TestMetrics(unittest.TestCase):

    def test_metrics(self):
        # numpy.random.seed(2334) # Fix seed

        tp = keras_metrics.true_positive()
        tn = keras_metrics.true_negative()
        fp = keras_metrics.false_positive()
        fn = keras_metrics.false_negative()

        precision = keras_metrics.precision()
        recall = keras_metrics.recall()
        f1 = keras_metrics.f1_score()

        model_fn = "./temp_model.hdf5"
        model = keras.models.Sequential()
        # model.add(keras.layers.Input((1,)))
        model.add(keras.layers.Activation(keras.backend.sin, input_shape=(1,)))
        model.add(keras.layers.Activation(keras.backend.abs))

        model.compile(optimizer="sgd",
                    loss="binary_crossentropy",
                        metrics=[tp, tn, fp, fn, precision, recall, f1])

        samples = 10000
        batch_size = 100
        lim = numpy.pi/2

        numpy.random.seed(2333)  # Fix seed
        x = numpy.random.uniform(0, lim, (samples, 1))
        y = numpy.random.randint(2, size=(samples, 1))

        if os.path.isfile(model_fn):
            print("Load saved model weights from %s" % model_fn)
            model.load_weights(model_fn)
        else:
            model.fit(x, y, epochs=10, batch_size=batch_size)
            model.save_weights(model_fn, overwrite=False)
            print("Svae model weights to %s" % model_fn)

        print("Evaluate model...")
        metrics = model.evaluate(x, y, batch_size=batch_size)[1:]
        y_pred = model.predict(x)

        metrics = list(map(float, metrics))

        tp_val = metrics[0]
        tn_val = metrics[1]
        fp_val = metrics[2]
        fn_val = metrics[3]

        precision = metrics[4]
        recall = metrics[5]
        f1 = metrics[6]

        expected_precision = tp_val / (tp_val + fp_val)
        expected_recall = tp_val / (tp_val + fn_val)

        f1_divident = (expected_precision*expected_recall)
        f1_divisor = (expected_precision+expected_recall)
        expected_f1 = (2 * f1_divident / f1_divisor)

        self.assertGreaterEqual(tp_val, 0.0)
        self.assertGreaterEqual(fp_val, 0.0)
        self.assertGreaterEqual(fn_val, 0.0)
        self.assertGreaterEqual(tn_val, 0.0)

        # Compare to numpy estimation
        expected_tp = ref_true_pos(y, y_pred)
        expected_fp = ref_false_pos(y, y_pred)
        expected_fn = ref_false_neg(y, y_pred)
        expected_tn = ref_true_neg(y, y_pred)

        self.assertEqual(tp_val, expected_tp)
        self.assertEqual(fp_val, expected_fp)
        self.assertEqual(fn_val, expected_fn)
        self.assertEqual(tn_val, expected_tn)

        # Check summation
        self.assertEqual(sum(metrics[0:4]), samples)

        places = 4
        self.assertAlmostEqual(expected_precision, precision, places=places)
        self.assertAlmostEqual(expected_recall, recall, places=places)
        self.assertAlmostEqual(expected_f1, f1, places=places)


if __name__ == "__main__":
    unittest.main()
