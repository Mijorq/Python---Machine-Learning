def SVC (Training_Data, Testing_Data, index):
    from sklearn.svm import SVC
    import pandas as pd
    import numpy as np
    from sklearn import utils as ut
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
    import time
    from skopt import BayesSearchCV
    from colorama import Fore, Style
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

    sv = SVC()

    param_dist = {'C': np.arange(1, 52, 10),
                  'kernel': ['sigmoid', 'rbf'],
                  'cache_size': [200, 300, 400],
                  'degree': np.arange(3, 8),
                  'gamma': ['scale', 'auto'],
                  'coef0': np.arange(0.001, 10, 0.5),
                  'tol': [1e-1, 1e-3, 1e-5],
                  'probability': [True, False],
                  }

    print (Fore.YELLOW + '\n SVM Results ' + str(index+1))
    print(Style.RESET_ALL)

    clf = BayesSearchCV(sv, param_dist, cv=10, n_jobs=-1)
    clf.fit(Training_X, Training_Y, callback=on_step)
    df = pd.DataFrame(clf.cv_results_)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print (df[['mean_test_score', 'params']])
    print(clf.best_params_)
    print(clf.best_score_)

    Pred_X = clf.predict(Testing_X)
    score = accuracy_score(Pred_X, Testing_Y)

    plot_confusion_matrix(clf, Testing_X, Testing_Y)
    (plt.gcf()).savefig(str(index+1) + '_SVC_Plot')
    plt.close()
    time.sleep(5)

    print (classification_report(Pred_X, Testing_Y))
    print (score)
    print(Fore.YELLOW + '\n End of SVM Results ' + str(index+1))
    print(Style.RESET_ALL)

    Best_result = [clf.best_score_,score,  clf.best_params_]
    print (Best_result)

    return Pred_X, Best_result
