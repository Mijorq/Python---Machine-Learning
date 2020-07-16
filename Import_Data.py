def Import_Data (address, worksheet):

    import pandas as pd


    Database = pd.read_excel(address, worksheet, index_col=0)
#    print ("Database shape is:")
#    print (Database.shape)

    Data = Database

    condition_dict = {'STABLE': 1, 'FAILURE': 2, 'MAJOR FAILURE': 3}
    Data['Condition'] = Data['Condition'].map(condition_dict)
    damage_dict = {'STABLE': 1, 'VERY LOW': 2, 'LOW': 3, 'HIGH': 4, 'VERY HIGH': 4}
    Data['Damage'] = Data['Damage'].map(damage_dict)
    Data = Data.drop(labels=['Mine', 'Name', 'Method', 'Wall', 'Origin', 'Inclination'], axis=1)

    return Data, Database
