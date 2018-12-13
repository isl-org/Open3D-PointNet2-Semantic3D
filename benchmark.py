import json
import numpy as np

import time

from predict import Predictor


if __name__ == "__main__":
    checkpoint = "logs/semantic_backup_full_submit_dec_10/best_model_epoch_275.ckpt"
    hyper_params = json.loads(open("semantic.json").read())
    predictor = Predictor(
        checkpoint_path=checkpoint, num_classes=9, hyper_params=hyper_params
    )

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
