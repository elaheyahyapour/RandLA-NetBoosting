import tensorflow as tf
import numpy as np
from helper_tool1 import ConfigToronto3D as cfg


def inspect_data(dataset, num_samples=5):
    dataset.init_train_pipeline()  # Initialize the data pipeline
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_samples):
            try:
                data = sess.run(dataset.flat_inputs)
                print("Sample {}:".format(i + 1))
                print("  XYZ shape: {}".format(data[0].shape))
                print("  Features shape: {}".format(data[1].shape))
                print("  Labels shape: {}".format(data[2].shape))
                print("  XYZ min/max: {}/{}".format(np.min(data[0]), np.max(data[0])))
                print("  Features min/max: {}/{}".format(np.min(data[1]), np.max(data[1])))
                print("  Unique labels: {}".format(np.unique(data[2])))
                print("  Any NaNs in XYZ: {}".format(np.isnan(data[0]).any()))
                print("  Any NaNs in Features: {}".format(np.isnan(data[1]).any()))

            except tf.errors.OutOfRangeError:
                print("Reached end of dataset")
                break



def verify_labels(dataset):
    dataset.init_train_pipeline()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while True:
                data = sess.run(dataset.flat_inputs)
                labels = data[2]
                unique_labels = np.unique(labels)
                if not np.all(np.isin(unique_labels, range(dataset.num_classes))):
                    print("Warning: Unexpected labels found: {}".format(unique_labels))
                    return False
        except tf.errors.OutOfRangeError:
            print("Finished verifying labels")
            return True


def check_class_distribution(dataset):
    dataset.init_train_pipeline()
    class_counts = np.zeros(dataset.num_classes, dtype=int)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while True:
                data = sess.run(dataset.flat_inputs)
                labels = data[2]
                unique, counts = np.unique(labels, return_counts=True)
                class_counts[unique] += counts
        except tf.errors.OutOfRangeError:
            pass

    for i, count in enumerate(class_counts):
        print("Class {}: {} samples".format(i, count))

    if 0 in class_counts:
        print("Warning: Some classes have zero samples!")



def check_feature_ranges(dataset):
    dataset.init_train_pipeline()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while True:
                data = sess.run(dataset.flat_inputs)
                features = data[1]
                if np.any(features < 0) or np.any(features > 1):  # Assuming normalized features
                    print("Warning: Features found outside the range [0, 1]")
                    return False
        except tf.errors.OutOfRangeError:
            print("All features are within the expected range")
            return True



def run_all_checks(dataset):
    print("Running comprehensive data checks...")
    inspect_data(dataset)
    verify_labels(dataset)
    check_class_distribution(dataset)
    check_feature_ranges(dataset)
    print("Comprehensive checks complete.")

if __name__ == "__main__":
    # Create the dataset
    dataset = Toronto3D(mode='train')

    # Initialize the dataset pipeline
    dataset.init_train_pipeline()

    # Run all checks
    run_all_checks(dataset)