import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# Classe per la diagnostica del modello
class ModelDiagnostics:
    def __init__(self, X, y, coefficients, channels, additional_columns, feature_names):
        self.X = X
        self.y = y
        self.coefficients = coefficients
        self.channels = channels
        self.additional_columns = additional_columns
        self.residuals = y - X @ coefficients
        self.n = len(y)  # Numero di osservazioni
        self.p = X.shape[1]  # Numero di parametri (incluso l'intercetta)
        self.feature_names = feature_names

    def calculate_metrics(self):
        """
        Calcola le metriche di diagnostica del modello.

        Returns:
            dict: Un dizionario contenente le metriche calcolate.
        """
        predictions = self.X @ self.coefficients
        r_squared = r2_score(self.y, predictions)
        adjusted_r_squared = self._adjusted_r_squared(r_squared)
        rmse = np.sqrt(mean_squared_error(self.y, predictions))
        dw = durbin_watson(self.residuals)
        mape = mean_absolute_percentage_error(self.y, predictions)

        return {
            "R²": r_squared,
            "Adjusted R²": adjusted_r_squared,
            "RMSE": rmse,
            "Durbin-Watson": dw,
            "MAPE": mape,
        }

    def _adjusted_r_squared(self, r_squared):
        """
        Calcola il valore di R² corretto.

        Args:
            r_squared (float): Il valore di R².

        Returns:
            float: Il valore di R² corretto.
        """
        n = len(self.y)
        return 1 - (1 - r_squared) * (n - 1) / (n - self.p - 1)

    def calculate_regression_summary(self):
        """
        Calcola un riepilogo della regressione contenente
        il nome del coefficiente, il valore del coefficiente, il p-value,
        l'intervallo di confidenza superiore e inferiore al 95%.

        Returns:
            DataFrame: Tabella con coefficiente, p-value, upper
            e lower bounds.
        """
        residual_variance = np.sum(self.residuals**2) / (self.n - self.p)
        cov_matrix = self._calculate_covariance_matrix(residual_variance)

        std_errors = np.sqrt(np.diag(cov_matrix))
        t_stats = self.coefficients / std_errors
        p_values = self._calculate_p_values(t_stats)

        lower_bound, upper_bound = self._confidence_intervals(std_errors)

        summary_df = pd.DataFrame(
            {
                "Coefficient Name": self.feature_names,
                "Coefficient Value": self.coefficients,
                "Standard Error": std_errors,
                "P-Value": p_values,
                "Lower 95% CL": lower_bound,
                "Upper 95% CL": upper_bound,
            }
        )

        return summary_df

    def _calculate_covariance_matrix(self, residual_variance):
        """
        Calcola la matrice di covarianza dei coefficienti.

        Args:
            residual_variance (float): Varianza dei residui.

        Returns:
            ndarray: La matrice di covarianza.
        """
        XTX_inv = np.linalg.inv(self.X.T @ self.X)
        return residual_variance * XTX_inv

    def _calculate_p_values(self, t_stats):
        """
        Calcola i p-value dai t-statistics.

        Args:
            t_stats (ndarray): I t-statistics.

        Returns:
            ndarray: I p-value calcolati.
        """
        return 2 * (1 - stats.t.cdf(np.abs(t_stats), df=self.n - self.p))

    def _confidence_intervals(self, std_errors):
        """
        Calcola gli intervalli di confidenza al 95%.

        Args:
            std_errors (ndarray): Gli errori standard dei coefficienti.

        Returns:
            tuple: Limite inferiore e superiore degli intervalli di confidenza.
        """
        confidence_interval_95 = 1.96 * std_errors
        lower_bound = self.coefficients - confidence_interval_95
        upper_bound = self.coefficients + confidence_interval_95
        return lower_bound, upper_bound

    def plot_acf(self):
        """
        Plot dell'ACF dei residui.

        Returns:
            ndarray: I valori ACF dei residui.
        """
        acf_values = acf(self.residuals, fft=False)
        plt.figure(figsize=(10, 6))
        plt.stem(range(len(acf_values)), acf_values, basefmt=" ")
        plt.title("ACF dei Residui")
        plt.xlabel("Lags")
        plt.ylabel("Autocorrelation")
        plt.show()
        return acf_values
