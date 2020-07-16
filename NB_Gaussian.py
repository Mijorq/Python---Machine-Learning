def NB_G (Training_Data, Testing_Data, index):
    import pandas as pd
    from sklearn import utils as ut
    import numpy as np
    import time
    from colorama import Fore, Style

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

    from sklearn.naive_bayes import GaussianNB
    nbg = GaussianNB()

    from skopt import BayesSearchCV
    import pandas as pd
    priors = np.asarray([0.5, 0.3, 0.2])
    param_dist = {"var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-3, 1e-2, 1e-1],
#                  "priors": priors
                  }

    print (Fore.LIGHTMAGENTA_EX + '\n Naive-Bayes Gaussian Results ' + str(index+1))
    print(Style.RESET_ALL)

    clf = BayesSearchCV(nbg, param_dist, cv=10, n_jobs=-1)
    clf.fit(Training_X, Training_Y, callback=on_step)
    df = pd.DataFrame(clf.cv_results_)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print (df[['mean_test_score', 'params']])
    print(clf.best_params_)
    print(clf.best_score_)

    Pred_X = clf.predict(Testing_X)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
    import matplotlib.pyplot as plt

    score = accuracy_score(Pred_X, Testing_Y)

    plot_confusion_matrix(clf, Testing_X, Testing_Y)
    (plt.gcf()).savefig(str(index+1) + '_NB_G_Plot')
    plt.close()
    time.sleep(5)

    print (classification_report(Pred_X, Testing_Y))
    print (score)
    print(Fore.LIGHTMAGENTA_EX + '\n End of Naive-Bayes Gaussian Results ' + str(index+1))
    print(Style.RESET_ALL)

    Best_result = [clf.best_score_,score,  clf.best_params_]
    print (Best_result)

    return Pred_X, Best_result
