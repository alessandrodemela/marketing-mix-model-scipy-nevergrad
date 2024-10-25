import numpy as np

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# Classe per la trasformazione dei dati
class DataTransformer:
    def __init__(self):
        pass

    def calculate_adstock(self, x, decay_rate):
        """
        Calcola l'adstock con un tasso di decay specifico.
        """

        if len(x) == 0:
            return x  # Return empty array as is
    
        x=np.array(x, dtype=float)
        adstock = np.zeros_like(x, dtype=float)
        adstock[0] = x[0]
        for i in range(1, len(x)):
            adstock[i] = x[i] + decay_rate * adstock[i - 1]
        return adstock

    def sigmoid(self, x, k, x0):
        """
        Funzione sigmoidale per la saturazione.
        """
        return 1 / (1 + np.exp(-k * (x - x0))) * x

    def hill(self, x, k, x0):
        """
        Funzione hill per la saturazione.
        Params:
            max_response: risposta massima possibile
            x: input values
            k: parametro di forma
            x0: punto di metà saturazione
        """
        return (x**k) / (x0**k + x**k)

    def apply_saturation(self, x, saturation_type, k, x0):
        """
        Applica la saturazione ai dati.
        """
        if saturation_type == "sigmoid":
            return self.sigmoid(x, k, x0) * x
        elif saturation_type == "hill":
            return self.hill(x, k, x0) * x
        else:
            raise ValueError("Tipo di saturazione non supportato")

    def apply_transformations(self, data, channel, decay_rate,
                              saturation_type, k, x0):
        """
        Applica la trasformazione adstock e saturazione per un canale.
        """
        adstock = self.calculate_adstock(data[channel].values, decay_rate)
        saturation = self.apply_saturation(adstock, saturation_type, k, x0)
        return (adstock, saturation)
