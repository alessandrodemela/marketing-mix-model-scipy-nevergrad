import numpy as np
import yaml
import nevergrad as ng
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_percentage_error

from data_transformer import DataTransformer

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# Classe per l'ottimizzazione del modello
class ModelOptimizer:
    def __init__(self, config_path, budget=1000):
        # Carica la configurazione dal file YAML
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # Estrai il target
        self.target = self.config["target"]

        # Estrai i canali attivi e le colonne aggiuntive
        self.channels = [
            chan["name"] for chan in self.config["channels"] if chan["is_active"]
        ]
        self.additional_columns = [
            col["name"] for col in self.config["additional_columns"] if col["is_active"]
        ]

        print(
            f"Selected Target {self.target}\n"
            + "Media Channels {self.channels}\n"
            + "Additional Information {self.additional_columns}"
        )

        # Analizza i bounds
        self.bounds = self.parse_bounds()

        # Costruisci lo spazio dei parametri
        self.param_space = self.build_param_space()

        self.coefficients = None
        self.X = None
        self.y = None
        self.transformer = DataTransformer()

        self.budget = budget

        self.params = None

    def parse_bounds(self):
        """
        Analizza i bounds da configurazione e restituisce una lista di bounds.
        """

        def parse_value(value):
            if value == "None":
                return None
            elif value == "-inf":
                return -np.inf
            elif value == "inf":
                return np.inf
            else:
                return float(value)

        bounds = []
        for chan in (
            [self.config["baseline"]]
            + self.config["channels"]
            + self.config["additional_columns"]
        ):
            if not chan["is_active"]:
                continue
            bounds.append([parse_value(b) for b in chan["bounds"]])
        return bounds

    def build_param_space(self):
        """
        Costruisce lo spazio dei parametri basato sulla configurazione YAML.
        """
        param_space = {}
        for channel in self.config["channels"]:
            if channel["is_active"]:
                param_space[f'{channel["name"]}_decay'] = ng.p.Scalar(
                    lower=channel["decay"]["lower"],
                    upper=channel["decay"]["upper"],
                    init=channel["decay"]["init"],
                )
                param_space[f'{channel["name"]}_saturation_type'] = ng.p.Choice(
                    [channel["saturation_type"]]
                )
                param_space[f'{channel["name"]}_k'] = ng.p.Log(
                    lower=channel["k"]["lower"],
                    upper=channel["k"]["upper"],
                )
                param_space[f'{channel["name"]}_x0'] = ng.p.Log(
                    lower=channel["x0"]["lower"],
                    upper=channel["x0"]["upper"],
                )

        return param_space

    def prepare_data(self, data, params):
        """
        Prepara i dati applicando le trasformazioni
        adstock e saturazione a ogni canale.
        """
        transformed_data = data.copy()
        for channel in self.channels:
            transformed_data[
                f"{channel}_transformed"
            ] = self.transformer.apply_transformations(
                data,
                channel,
                params[f"{channel}_decay"],
                params[f"{channel}_saturation_type"],
                params[f"{channel}_k"],
                params[f"{channel}_x0"],
            )
        return transformed_data

    def objective_function(self, params, data, init=None):
        """
        Funzione obiettivo che minimizza la somma dei quadrati
        degli errori (SSE).
        """
        transformed_data = self.prepare_data(data, params)
        self.X = transformed_data[
            [f"{channel}_transformed" for channel in self.channels]
            + self.additional_columns
        ].values
        self.y = transformed_data[self.target].values
        self.X = np.column_stack((np.ones(self.X.shape[0]), self.X))

        initial_coeffs = init if init is not None else np.zeros(self.X.shape[1])

        def sse(coeffs):
            predictions = self.X @ coeffs
            return np.sum((self.y - predictions) ** 2)

        result = minimize(sse, initial_coeffs, bounds=self.bounds)
        self.coefficients = result.x

        return mean_absolute_percentage_error(self.y, self.X @ self.coefficients)

    def optimize_parameters(self, data, init=None):
        """
        Ottimizza i parametri con il metodo CMA di Nevergrad.
        """
        optimizer = ng.optimizers.CMA(
            parametrization=ng.p.Dict(**self.param_space), budget=self.budget
        )
        best_params = optimizer.minimize(
            lambda x: self.objective_function(x, data, init)
        )

        self.params = best_params
