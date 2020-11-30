import pandas as pd
import numpy as np
import os

file_path = os.path.dirname(__file__)


class production_data:
    def __init__(self, dmu, x, y, b=None, z=None):
        self.decision_making_unit = dmu
        self.x = x
        self.y = y
        self.b = b
        self.z = z


def load_GHG_abatement_cost(year=None, x_select=['HRSN', 'CPNK'], y_select=['VALK'], b_select=['GHG']):
    dataframe = pd.read_csv(
        file_path+"/data/abatementCost.csv", error_bad_lines=True)
    if year != None:
        dataframe = dataframe[dataframe['Year'] == year]
    else:
        dataframe['Country'] = dataframe['Country'] + \
            dataframe['Year'].apply(str)
    dmu = np.asmatrix(dataframe['Country']).T
    x = np.concatenate(
        [np.asmatrix(dataframe[selected]).T for selected in x_select], axis=1)
    y = np.concatenate(
        [np.asmatrix(dataframe[selected]).T for selected in y_select], axis=1)
    if b_select != None:
        b = np.concatenate(
            [np.asmatrix(dataframe[selected]).T for selected in b_select], axis=1)
    return production_data(dmu, x, y, b)


def load_Finnish_electricity_firm(x_select=['Energy', 'Length', 'Customers'], y_select=['OPEX', 'CAPEX', 'TOTEX'], z_select=['PerUndGr']):
    dataframe = pd.read_csv(
        file_path+"/data/electricityFirms.csv", error_bad_lines=True)
    dmu = np.asmatrix(dataframe.index.tolist()).T
    x = np.concatenate(
        [np.asmatrix(dataframe[selected]).T for selected in x_select], axis=1)
    y = np.concatenate(
        [np.asmatrix(dataframe[selected]).T for selected in y_select], axis=1)
    if z_select != None:
        z = np.concatenate(
            [np.asmatrix(dataframe[selected]).T for selected in z_select], axis=1)
    return production_data(dmu, x, y, z=z)


def load_Tim_Coelli_frontier(x_select=['capital', 'labour'], y_select=['output']):
    dataframe = pd.read_csv(
        file_path+"/data/41Firm.csv", error_bad_lines=True)
    dmu = np.asmatrix(dataframe['firm']).T
    x = np.concatenate(
        [np.asmatrix(dataframe[selected]).T for selected in x_select], axis=1)
    y = np.concatenate(
        [np.asmatrix(dataframe[selected]).T for selected in y_select], axis=1)
    return production_data(dmu, x, y)


def load_Philipines_rice_production(year=None, x_select=['AREA', 'LABOR', 'NPK', 'OTHER', 'AREAP', 'LABORP', 'NPKP', 'OTHERP'], y_select=['PROD', 'PRICE']):
    dataframe = pd.read_csv(
        file_path+"/data/riceProduction.csv", error_bad_lines=True)
    if year != None:
        print(dataframe['YEARDUM'])
        dataframe = dataframe[dataframe['YEARDUM'] == year-1989]
    else:
        dataframe['FMERCODE'] = dataframe['FMERCODE'].apply(
            str) + ": " + (dataframe['YEARDUM'].apply(int)+1989).apply(str)
    dmu = np.asmatrix(dataframe['FMERCODE']).T
    x = np.concatenate(
        [np.asmatrix(dataframe[selected]).T for selected in x_select], axis=1)
    y = np.concatenate(
        [np.asmatrix(dataframe[selected]).T for selected in y_select], axis=1)
    return production_data(dmu, x, y)
