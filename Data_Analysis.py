def DA (Training_Data, Testing_Data):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import utils as ut

    df = Training_Data[['Rh', "Q'", 'A', 'B', 'C', "N'", 'Fail %', 'Damage']]
    pd.options.display.float_format = '{:,.3f}'.format
    correlation = df.corr()
    plt.figure(figsize=(16 ,10))
    sns.heatmap(correlation, annot=True)
    plt.show()

    df = Testing_Data[['Rh', "Q'", 'A', 'B', 'C', "N'", 'Fail %', 'Damage']]
    pd.options.display.float_format = '{:,.3f}'.format
    correlation = df.corr()
    plt.figure(figsize=(16 ,10))
    sns.heatmap(correlation, annot=True)
    plt.show()

    Training_X = Training_Data.drop(["Condition"], axis=1).values
    Training_X = ut.as_float_array(Training_X)

    Training_Y = Training_Data["Condition"].values
    Training_Y = ut.as_float_array(Training_Y)

    Testing_X = Testing_Data.drop(["Condition"], axis=1).values
    Testing_X = ut.as_float_array(Testing_X)

    Testing_Y = Testing_Data["Condition"].values
    Testing_Y = ut.as_float_array(Testing_Y)

    rf = RandomForestClassifier(random_state=0)
    rf.fit(Training_X, Training_Y)

    pd.options.display.float_format = '{:,.3f}'.format
    importance = rf.feature_importances_
    print ("Feature importance is:")
    print (importance)