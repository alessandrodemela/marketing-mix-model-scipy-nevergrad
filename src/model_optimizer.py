import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import nevergrad as ng
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_percentage_error

from data_transformer import DataTransformer
from model_diagnostics import ModelDiagnostics

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

        self.feature_names = ['Baseline'] + self.channels + self.additional_columns

        self.print_infos()

        # Costruisci lo spazio dei parametri
        self.param_space = self.build_param_space()

        self.coefficients = None
        self.coefficients_dict = None
        self.X = None
        self.y = None
        self.transformer = DataTransformer()

        self.budget = budget

        self.params = None
    
    def print_infos(self):
        print(
            f"\n{'Selected Target':25} {self.target}\n"
            + f"{'Media Channels':25} {self.channels}\n"
            + f"{'Additional Information':25} {self.additional_columns}\n"
        )

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
        self.coefficients_dict = {
            col : val 
            for col,val in zip(self.feature_names, result.x)
        }

        return mean_absolute_percentage_error(self.y, self.X @ self.coefficients)

    def optimize_parameters(self, data, init=None, start_date=None, end_date=None):
        """
        Ottimizza i parametri con il metodo CMA di Nevergrad.
        
        Parameters:
        - data: DataFrame containing the data for optimization.
        - init: Optional initial parameters for optimization.
        - start_date: Optional start date to filter the data.
        - end_date: Optional end date to filter the data.
        """
        
        # Filter data based on the specified date range
        if start_date is not None and end_date is not None:
            data = data.loc[start_date:end_date]
        elif start_date is not None:
            data = data.loc[start_date:]
        elif end_date is not None:
            data = data.loc[:end_date]
    
        # Drop features (columns) that are entirely zero
        zero_columns = data.columns[(data == 0).all()]
        
        for zero in zero_columns:
            for chan in self.config["channels"]:
                if chan["name"] == zero: chan["is_active"] = False
            for chan in self.config["additional_columns"]:
                if chan["name"] == zero: chan["is_active"] = False

        if len(zero_columns):
            self.channels = [
                chan["name"] for chan in self.config["channels"]
                if chan["is_active"]
            ]
            self.additional_columns = [
                col["name"] for col in self.config["additional_columns"]
                if col["is_active"]
            ]
            self.feature_names = ['Baseline'] + self.channels + self.additional_columns
        
        # Analizza i bounds
        self.bounds = self.parse_bounds()

        # Update param_space to remove zero-valued features
        self.param_space = {
            k: v for k, v in self.param_space.items() if k not in zero_columns}

        print(f'\nRemoved Feature(s) {zero_columns.to_list()}')
        self.print_infos()

        # Update the index date for optimization
        self.index_date = data.index

        optimizer = ng.optimizers.CMA(
            parametrization=ng.p.Dict(**self.param_space), budget=self.budget
        )
        try:
            best_params = optimizer.minimize(
                lambda x: self.objective_function(x, data, init)
            )
            self.params = best_params
            print('Optimization Succeded')
        except Exception as e:
            print('Optimization Failed', e)

    def get_diagnostics(self):
        """
        Calcola le metriche diagnostiche del modello.

        Returns:
            ModelDiagnostics: Un'istanza della classe ModelDiagnostics
            contenente le metriche calcolate.
        """
        diagnostics = ModelDiagnostics(
            self.X, self.y, self.coefficients, self.channels, 
            self.additional_columns, self.feature_names
            )
        return diagnostics
    
    def plot_variable_contributions(self):
        """
        Plotta i contributi delle singole variabili.
        """
        if self.coefficients is None:
            print("Coefficients are not calculated. Run the optimization first.")
            return

        # Calcola i contributi delle variabili
        contributions = (self.X * self.coefficients).sum(axis=0) / self.y.sum()
        print(contributions, self.feature_names)
        
        # Crea un DataFrame per le variabili e i loro contributi
        variable_contributions_dict = {
            'Variable': self.feature_names,
            'Contribution': contributions * 100
        }
        
        # Converti in DataFrame per il plotting
        contributions_df = pd.DataFrame(variable_contributions_dict)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.barh(contributions_df['Variable'], contributions_df['Contribution'], color='skyblue')
        plt.xlabel('Contributo')
        plt.title('Contributi delle Singole Variabili')
        plt.axvline(0, color='grey', linewidth=0.8, linestyle='--')
        plt.show()

    def plot_historical_kpi_with_contributions(self):
        """
        Plotta la serie storica del KPI con i contributi.
        
        Args:
            historical_data (DataFrame): I dati storici con la colonna KPI.
        """
        if self.coefficients is None:
            print("Coefficients are not calculated. Run the optimization first.")
            return

        # Calcola i contributi
        contributions = self.X * self.coefficients
        
        # Ottieni il KPI originale
        kpi = self.y
        
        # Plot
        plt.figure(figsize=(14, 7))

        # KPI in linea
        plt.subplot(1, 1, 1)
        plt.plot(self.index_date, kpi, label='KPI Originale', color='blue')
        plt.xlabel('Data')
        # Contributi in barre

        bottom = 0
        for contr, mkt_contr in zip(contributions.T, self.feature_names):
            plt.bar(
                self.index_date, contr, label=mkt_contr,
                bottom=bottom, width=5)
            bottom += contr

        plt.legend()
        plt.tight_layout()
        plt.show()
        
