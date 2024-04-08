import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Logger(object):
    """
        This class records certain variables to save them to a csv file.
        It is used to input data to a logger variable inspector class.
    """
    def __init__(self, names: list[str], filename: str):
        self.names = names
        self.data = [[] for _ in range(len(names))]
        self.x_data = [[] for _ in range(len(names))]
        self.filename = filename

    def addData(self, data: list[float], name: str):
        """
            Add data to the logger
        """
        if name not in self.names:
            raise Exception("Name not in logger")
        if len(data) != len(self.names):
            raise Exception("Data length does not match logger length")
        self.data[self.names.index(name)].append(data)
        self.x_data[self.names.index(name)].append(len(self.data[self.names.index(name)]))

    def saveData(self):
        # build dataframe, add timesteps as first column
        df = pd.DataFrame()
        df['timestep'] = self.x_data[0]
        for i in range(len(self.names)):
            df[self.names[i]] = self.data[i]

        # save to csv
        df.to_csv(self.filename, index=False)

