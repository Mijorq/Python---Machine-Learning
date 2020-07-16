def RF_Cross (Training_Data, Testing_Data, index):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    from sklearn import utils as ut
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
    from colorama import Fore, Style
    import time
    from sklearn.model_selection import cross_val_score
    import matplotlib.pyplot as plt

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

    cross_scores = np.mean(cross_val_score(rf, Training_X, Training_Y, cv=10))

    #Testing
    Pred_X = rf.predict(Testing_X)

    print (Fore.CYAN + '\n Random Forest Results ')
    print(Style.RESET_ALL)

    Pred_X = rf.predict(Testing_X)

    score = accuracy_score(Pred_X, Testing_Y)

    plot_confusion_matrix(rf, Testing_X, Testing_Y)
    (plt.gcf()).savefig(str(index+1) + '_RF_Cross_Plot')
    plt.close()
    time.sleep(5)

    print (classification_report(Pred_X, Testing_Y))
    print ("Cross Val Score " + str(cross_scores))
    print ("Accuracy Score " + str(score))
    print(Fore.CYAN + '\n End of Random Forest Results ')
    print (Style.RESET_ALL)

    return Pred_X, cross_scores, score
