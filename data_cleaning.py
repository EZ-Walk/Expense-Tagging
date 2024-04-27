# Take the data from the data/raw folder, clean it and save it to data/clean
# Cleaning steps:

import pandas as pd
import os
import numpy as np

pd.set_option('display.max_columns', 20)


if __name__ == '__main__':
    # 1. Read in the files
    print('Data cleaning')
    print('Current working directory:', os.getcwd())
    raw_files = os.listdir('data/raw')
    print(raw_files)
    
    raw_data = pd.DataFrame()
    for file in raw_files:
        raw_data = raw_data.append(pd.read_csv('data/raw/' + file), ignore_index=True)
    print('Data shape:', raw_data.shape)
    print(raw_data.head())
    
    # 2. keep only rows with Debit!=NaN, Status=Posted, and Account Number == 4430913
    raw_data = raw_data[raw_data['Debit'].notnull()]
    raw_data = raw_data[raw_data['Status'] == 'Posted']
    raw_data = raw_data[raw_data['Account Number'] == 443091309]
    print('Data shape:', raw_data.shape)
    

    # 3. keep only the desired columns
    data = raw_data[['Post Date', 'Description', 'Debit']]
    print('Data shape:', data.shape)
    
    # 4. convert the 'Post Date' column to datetime
    data['Post Date'] = data['Post Date'].astype('datetime64')
    
    # 5. Ranme the Post Date column to Date and the Debit column to Amount
    data.rename(columns={'Post Date': 'Date', 'Debit': 'Amount'}, inplace=True)

    # 5. Print the shape one last time and the date range encomapssed by the data as a Month, Day, Year
    print('Data shape:', data.shape)
    date_range = data['Date'].agg([np.min, np.max])
    print('Date range:', date_range[0].strftime('%B %d, %Y'), 'to', date_range[1].strftime('%B %d, %Y'))

    # 6. Save the data to data/clean/transactions.csv
    data.to_csv('data/clean/transactions.csv', index=False)