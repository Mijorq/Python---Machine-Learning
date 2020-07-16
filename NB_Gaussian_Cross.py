def NB_G_Cross (Training_Data, Testing_Data, index):
    import pandas as pd
    from sklearn import utils as ut
    import numpy as np
    import time
    from colorama import Fore, Style
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
    from sklearn.naive_bayes import GaussianNB
    import matplotlib.pyplot as plt

    Training_X = Training_Data.drop(["Condition"], axis=1).values
    Training_X = ut.as_float_array(Training_X)

    Training_Y = Training_Data["Condition"].values
    Training_Y = ut.as_float_array(Training_Y)

    Testing_X = Testing_Data.drop(["Condition"], axis=1).values
    Testing_X = ut.as_float_array(Testing_X)

    Testing_Y = Testing_Data["Condition"].values
    Testing_Y = ut.as_float_array(Testing_Y)

    nbg = GaussianNB()
    nbg.fit(Training_X, Training_Y)

    print (Fore.LIGHTMAGENTA_EX + '\n Naive-Bayes Gaussian Results ')
    print(Style.RESET_ALL)

    cross_scores = np.mean(cross_val_score(nbg, Training_X, Training_Y, cv=10))

    Pred_X = nbg.predict(Testing_X)

    score = accuracy_score(Pred_X, Testing_Y)

    plot_confusion_matrix(nbg, Testing_X, Testing_Y)
    (plt.gcf()).savefig(str(index+1) + '_NBG_Cross_Plot')
    plt.close()
    time.sleep(5)

    print (classification_report(Pred_X, Testing_Y))
    print ("Cross Val Score " + str(cross_scores))
    print ("Accuracy Score " + str(score))
    print(Fore.LIGHTMAGENTA_EX + '\n End of Naive-Bayes Gaussian Results ')
    print(Style.RESET_ALL)

    return Pred_X, cross_scores, score