import json
import numpy as np
import tensorflow as tf
import time

from predict import Predictor


if __name__ == "__main__":
    checkpoint = "logs/semantic_backup_full_submit_dec_10/best_model_epoch_275.ckpt"
    hyper_params = json.loads(open("semantic.json").read())
    predictor = Predictor(
        checkpoint_path=checkpoint, num_classes=9, hyper_params=hyper_params
    )

    batch_size = 64
    # Init data
    points_with_colors = np.random.randn(batch_size, hyper_params["num_point"], 6)

    # Warm up
    pd_labels = predictor.predict(points_with_colors)

    # Benchmark
    s = time.time()

    profiler = tf.profiler.Profiler(predictor.sess.graph)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    _ = predictor.predict(
        points_with_colors, run_options=run_options, run_metadata=run_metadata
    )

    profiler.add_step(0, run_metadata)

    batch_time = time.time() - s
    sample_time = batch_time / batch_size
    print(
        "Batch size: {}, batch_time: {}, sample_time: {}".format(
            batch_size, batch_time, sample_time
        )
    )

    option_builder = tf.profiler.ProfileOptionBuilder
    opts = (
        option_builder(option_builder.time_and_memory())
        .with_step(-1)  # with -1, should compute the average of all registered steps.
        .with_file_output("tf-profile.txt")
        .select(["micros", "bytes", "occurrence"])
        .order_by("micros")
        .build()
    )
    # Profiling info about ops are saved in 'test-%s.txt' % FLAGS.out
    profiler.profile_operations(options=opts)

    for batch_size in [2 ** n for n in range(8)]:
        # Init data
        points_with_colors = np.random.randn(batch_size, hyper_params["num_point"], 6)

        # Warm up
        pd_labels = predictor.predict(points_with_colors)

        # Benchmark
        s = time.time()
        _ = predictor.predict(points_with_colors)
        batch_time = time.time() - s
        sample_time = batch_time / batch_size
        print(
            "Batch size: {}, batch_time: {}, sample_time: {}".format(
                batch_size, batch_time, sample_time
            )
        )
