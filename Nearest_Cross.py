def KNB_Cross(Training_Data, Testing_Data, index):
    import pandas as pd
    from sklearn import utils as ut
    import numpy as np
    import time
    from colorama import Fore, Style
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
    import matplotlib.pyplot as plt

    Training_X = Training_Data.drop(["Condition"], axis=1).values
    Training_X = ut.as_float_array(Training_X)

    Training_Y = Training_Data["Condition"].values
    Training_Y = ut.as_float_array(Training_Y)

    Testing_X = Testing_Data.drop(["Condition"], axis=1).values
    Testing_X = ut.as_float_array(Testing_X)

    Testing_Y = Testing_Data["Condition"].values
    Testing_Y = ut.as_float_array(Testing_Y)

    knb = KNeighborsClassifier ()
    knb.fit(Training_X, Training_Y)

    print(Fore.LIGHTMAGENTA_EX + '\n KNN Results ')
    print(Style.RESET_ALL)

    cross_scores = np.mean(cross_val_score(knb, Training_X, Training_Y, cv=10))

    Pred_X = knb.predict(Testing_X)

    score = accuracy_score(Pred_X, Testing_Y)

    plot_confusion_matrix(knb, Testing_X, Testing_Y)
    (plt.gcf()).savefig(str(index+1) + '_KNN_Cross_Plot')
    plt.close()
    time.sleep(5)

    print(classification_report(Pred_X, Testing_Y))
    print("Cross Val Score " + str(cross_scores))
    print("Accuracy Score " + str(score))
    print(Fore.LIGHTMAGENTA_EX + '\n End of KNN Results ')
    print(Style.RESET_ALL)

    return Pred_X, cross_scores, score