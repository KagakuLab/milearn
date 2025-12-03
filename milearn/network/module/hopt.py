import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch

DEFAULT_PARAM_GRID = {

    # Fixed hparams
    "max_epochs": 1000,
    "early_stopping": True,
    "accelerator": "cpu",
    "random_seed": 42,
    "verbose": False,

    # Activation function
    "activation": ["relu", "leakyrelu", "gelu", "elu", "silu"],

    # Learning dynamics
    "learning_rate": [10e-5, 10e-4],
    "batch_size": [32, 512, 1024],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3, 1e-2],

    # MIL specific
    "tau": [0.01, 0.5, 1.0],
    "instance_dropout": [0.0, 0.2, 0.4, 0.6, 0.8],

    # Architecture depth/shape
    "hidden_layer_sizes": [(2048, 1024, 512, 256, 128, 64), (256, 128, 64), (128,)],

}

def get_optimal_torch_threads(n_jobs: int) -> int:
    """Get optimal number of torch threads for parallel jobs.

    Args:
        n_jobs (int): number of parallel jobs.

    Returns:
        int: number of threads to allocate per job.
    """
    total_cpus = os.cpu_count() or 1
    return max(1, total_cpus // n_jobs)


class StepwiseHopt:
    """Stepwise hyperparameter optimization."""

    def _evaluate_model(self, cls, hparams, best_params, param, val, x, y, n_jobs):
        """Train and evaluate a model for a given hyperparameter setting.

        Args:
            cls (type): model class.
            hparams (dict): base hyperparameters.
            best_params (dict): best parameters found so far.
            param (str): name of the parameter being optimized.
            val (Any): value of the parameter being tested.
            x (list | np.ndarray): input data.
            y (list | np.ndarray): target labels.
            n_jobs (int): number of parallel jobs.

        Returns:
            tuple: (val, loss, epochs_trained, elapsed_model_time).
        """
        torch.set_num_threads(get_optimal_torch_threads(n_jobs))
        valid_args = set(hparams.keys())
        tmp_params = {**hparams, **best_params, param: val}
        safe_params = {k: v for k, v in tmp_params.items() if k in valid_args}

        tmp_model = cls(**safe_params)

        start_model_time = time.time()
        tmp_model.fit(x, y)
        elapsed_model_time = time.time() - start_model_time

        epochs_trained = tmp_model._trainer.current_epoch + 1
        loss = float(tmp_model._trainer.callback_metrics["val_loss"])
        return val, loss, epochs_trained, elapsed_model_time

    def hopt(self, x, y, param_grid=None, verbose=True):
        """Perform stepwise hyperparameter optimization.

        Args:
            x (list | np.ndarray): input data.
            y (list | np.ndarray): target labels.
            param_grid (dict | None): parameter grid to search.
            verbose (bool): whether to print progress.

        Returns:
            dict: best hyperparameters found.
        """
        if param_grid is None:
            param_grid = DEFAULT_PARAM_GRID

        valid_args = set(self.hparams.keys())
        filtered_grid = {k: v for k, v in param_grid.items() if k in valid_args}

        total_steps = sum(len(v) for v in filtered_grid.values() if isinstance(v, (list, tuple)))
        current_step = 0
        start_time = time.time()

        best_params = {}
        for param, options in filtered_grid.items():
            if not isinstance(options, (list)):  # fixed hyperparameter
                best_params[param] = options
                continue

            best_val = None
            best_loss = float("inf")

            if verbose:
                print(f"Optimizing hyperparameter: {param} ({len(options)} options)")

            n_jobs = len(options)
            args_list = [(self.__class__, self.hparams, best_params, param, val, x, y, n_jobs) for val in options]

            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(lambda args: self._evaluate_model(*args), args_list))

            for val, loss, epochs, model_time in results:
                current_step += 1
                progress_pct = (current_step / total_steps) * 100
                elapsed_min = model_time / 60.0

                if verbose:
                    print(
                        f"[{current_step}/{total_steps} | {progress_pct:4.1f}% | {elapsed_min:4.1f} min] "
                        f"Value: {str(val)}, Epochs: {epochs}, Loss: {loss:.4f}"
                    )

                if loss < best_loss:
                    best_loss = loss
                    best_val = val

            best_params[param] = best_val

            if verbose and best_val is not None:
                print(f"Best {param} = {str(best_val)}, val_loss = {best_loss:.4f}")

        self.hparams.update(best_params)

        total_time_min = (time.time() - start_time) / 60.0
        if verbose:
            print(f"Stepwise optimization completed in {total_time_min:.1f} min\n")

        return best_params
