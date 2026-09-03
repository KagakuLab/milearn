
milearn: Multi-instance machine learning in Python
==========================================================
``milearn`` is designed to mimic the scikit-learn interface to simplify its usage and integration with other tools.

Key Features
------------------------

- Traditional and neural network-based MIL algorithms (regression and classification)
- Integrated stepwise model hyperparameter optimization (recommended for small datasets)

Installation
------------------------

.. code-block:: bash

    pip install mikit-learn

Quick Start
------------------------

.. code-block:: python

    from milearn.data.mnist import load_mnist, create_bags_reg
    from milearn.preprocessing import BagMinMaxScaler
    from sklearn.model_selection import train_test_split
    from milearn.network.module.hopt import DEFAULT_PARAM_GRID
    from milearn.network.regressor import DynamicPoolingNetworkRegressor

    # 1. Create MNIST regression dataset
    data, targets = load_mnist()
    bags, labels, key = create_bags_reg(data, targets, bag_size=10, num_bags=10000,
                                        bag_agg="mean", random_state=42)

    # 2. Train/test split and scale features
    x_train, x_test, y_train, y_test, key_train, key_test = train_test_split(bags, labels, key,
                                                                             random_state=42)
    scaler = BagMinMaxScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 3. Train model
    model = DynamicPoolingNetworkRegressor()
    model.hopt(x_train_scaled, y_train,  # recommended for small datasets only
               param_grid=DEFAULT_PARAM_GRID, verbose=True)
    model.fit(x_train_scaled, y_train)

    # 4. Get predictions
    y_pred = model.predict(x_test_scaled)  # predicted labels
    w_pred = model.get_instance_weights(x_test_scaled)  # predicted instance weights

Tutorials
------------------------

Several examples of the ``milearn`` application to the classification/regression problem and key instance detection
for the MNIST dataset can be found in `tutorial collection <notebooks>`_ .

Paper
------------------------

**Warning**: currently, notebooks for reproducing paper results are migrating between ``milearn`` and ``QSARmil``,
so some may be unavailable for some time.

Application cases demonstrated in the paper can be found in:

- `MNIST classification <https://github.com/KagakuLab/milearn/blob/main/notebooks/Tutorial_2_KID_for_mnist_classification.ipynb>`_
- `MNIST regression <https://github.com/KagakuLab/milearn/blob/main/notebooks/Tutorial_3_KID_for_mnist_regression.ipynb>`_
- `Molecular conformers <https://github.com/KagakuLab/QSARmil/blob/main/notebooks/03_Key_Instance_Detection.ipynb>`_
- `Molecular fragments <https://github.com/KagakuLab/QSARmil/blob/main/notebooks/Tutorial_3_KID_for_fragments.ipynb>`_
- `Protein-protein interaction <https://github.com/KagakuLab/SEQmil/blob/main/notebooks/Tutorial_1_KID_for_protein_protein_interaction.ipynb>`_

Citation
------------------------

For ``milearn`` citation use:

.. code-block:: bibtex

    @article{zankov2025milearn,
      title={milearn: A Python Package for Multi-Instance Machine Learning},
      author={Zankov, Dmitry and Polishchuk, Pavlo and Sobieraj, Michal and Barbatti, Mario},
      journal={arXiv preprint arXiv:2512.01287},
      year={2025}
    }


