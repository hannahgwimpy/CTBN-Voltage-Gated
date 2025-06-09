# CTBN-Voltage-Gated

[![DOI](https://zenodo.org/badge/998520127.svg)](https://doi.org/10.5281/zenodo.15625711)

A GUI application and associated scripts for simulating voltage-gated sodium channel models, showcasing CTBN Markov model's advantages over the Legacy Hodgkin-Huxley and Markov models. It showcases sodium current traces with customizable parameters and voltage protocols, allowing users to explore the differences between Hodgkin-Huxley, CTBN Markov, and legacy Markov models. The `data` directory contains data generated from model comparisons and figures used in the associated paper.

## Features

*   Simulates sodium current traces using three different models:
    *   Continuous-Time Bayesian Networks (CTBN) Markov
    *   Legacy Markov
    *   Hodgkin-Huxley
*   Allows customization of model parameters.
*   Provides a flexible interface to define custom voltage protocols.
*   Generates and displays sodium current traces.
*   Saves generated plots to the `data/currents/` directory.
*   Includes comparative data and figures from research in the `data/` directory.

## Installation

1.  **Prerequisites**:
    *   Python 3.8 or higher.
    *   Git (for cloning the repository).

2.  **Clone the repository**:
    ```bash
    git clone https://github.com/hannahgwimpy/CTBN-Voltage-Gated.git
    cd CTBN-Voltage-Gated
    ```

3.  **Install dependencies and the package**:
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    Then, install the project (this will also install dependencies from `pyproject.toml`):
    ```bash
    pip install .
    ```
    Alternatively, if you only want to install dependencies for development without installing the package itself, you can use the `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once installed, you can run the application using the command:

```bash
CTBN-Voltage-Gated
```

Alternatively, you can run the main script directly from the `src` directory (e.g., during development):

```bash
python src/main.py
```

**Using the GUI:**

1.  **Select a Model**: Choose between "CTBN Markov", "Legacy Markov", or "Hodgkin-Huxley" from the "Model Selection" dropdown.
2.  **Adjust Parameters**: Modify model-specific parameters under the "Model Parameters" section.
3.  **Define Voltage Protocol**: Customize the voltage protocol settings (Holding Potential, Durations, Test Voltages) under the "Voltage Protocol" section. Add or remove voltage steps as needed and click "Apply Protocol".
4.  **Run Simulation**: Click the "Run Simulation" button.
5.  **View Results**: The application will generate and display the sodium current trace. The plot will also be automatically saved to the `data/currents/` directory.

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -am 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Create a new Pull Request.

Please make sure to update tests as appropriate.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
