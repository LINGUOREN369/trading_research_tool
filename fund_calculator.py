## concat all of the csv files in the folder 
import pandas as pd
import os

## using folder Stock_History


def concat_csv_files(folder_path):
    """
    Concatenate all CSV files in the specified folder into a single DataFrame.
    
    Parameters:
    folder_path (str): The path to the folder containing the CSV files.
    
    Returns:
    pd.DataFrame: A DataFrame containing the concatenated data from all CSV files.
    """
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = []
    
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    return pd.concat(dataframes, ignore_index=True)



def calculate_dividends_and_funds(df):
    """
    Calculate total dividends and electric fund contributions.
    
    Parameters:
        df (pd.DataFrame): Stock transaction data.
    
    Returns:
        tuple: (total_dividends, total_electric_fund)
    """
    total_dividends = df[df["Action"].str.contains("DIVIDEND", case=False, na=False)]["Amount ($)"].sum()
    total_electric_fund = df[df["Action"].str.contains("Electronic Funds Transfer Received \(Cash\)", case=False, na=False)]["Amount ($)"].sum()
    
    return total_dividends, total_electric_fund






