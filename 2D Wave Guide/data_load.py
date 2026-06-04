import pandas as pd
import numpy as np

class WaveguideBoundaryData:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self.freqs = self.df['f'].unique()
        self.x = self.df['x'].iloc[0]
        
    def get_training_data(self):
        Y = {}
        U_re = {}
        U_im = {}

        for f in self.freqs:
            subset = self.df[self.df['f'] == f]
            f = str(f)
            Y[f] = np.array(subset['y'].values, dtype=np.float32)
            U_re[f] = np.array(subset['Re_U'].values, dtype=np.float32)
            U_im[f] = np.array(subset['Im_U'].values, dtype=np.float32)
        
        return self.x, Y, U_re, U_im, np.array(self.freqs, dtype=str)