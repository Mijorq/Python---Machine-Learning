def KNB (Training_Data, Testing_Data, index):
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    import numpy as np
    from sklearn import utils as ut
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
    from colorama import Fore, Style
    from skopt import BayesSearchCV
    import time
    import matplotlib.pyplot as plt

    def on_step(optim_result):
        score = clf.best_score_
        print("best score: %s" % score)
        if score >= 0.98:
            print('Interrupting!')
            return True

    Training_X = Training_Data.drop(["Condition"], axis=1).values
    Training_X = ut.as_float_array(Training_X)

    Training_Y = Training_Data["Condition"].values
    Training_Y = ut.as_float_array(Training_Y)

    Testing_X = Testing_Data.drop(["Condition"], axis=1).values
    Testing_X = ut.as_float_array(Testing_X)

    Testing_Y = Testing_Data["Condition"].values
    Testing_Y = ut.as_float_array(Testing_Y)

    knb = KNeighborsClassifier ()

    param_dist = {'n_neighbors': [2, 5, 7, 10],
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                  'leaf_size': [20, 30, 40],
                  'p': [1, 2, 3, 4],
                  'metric': ['euclidean', 'manhattan', 'minkowski']
                  }

    print (Fore.CYAN + '\n KNN Results ' + str(index+1))
    print(Style.RESET_ALL)
    clf = BayesSearchCV(knb, param_dist, cv=10, n_jobs=-1)
    clf.fit(Training_X, Training_Y, callback=on_step)
    df = pd.DataFrame(clf.cv_results_)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print (df[['mean_test_score', 'params']])
    print(clf.best_params_)
    print(clf.best_score_)

    Pred_X = clf.predict(Testing_X)

    score = accuracy_score(Pred_X, Testing_Y)

    plot_confusion_matrix(clf, Testing_X, Testing_Y)
    (plt.gcf()).savefig(str(index+1) + '_KNN_Plot')
    plt.close()
    time.sleep(5)

    print (classification_report(Pred_X, Testing_Y))
    print (score)
    print(Fore.CYAN + '\n End of KNN Results ' + str(index+1))
    print (Style.RESET_ALL)

    Best_result = [clf.best_score_,score,  clf.best_params_]

    return Pred_X, Best_result