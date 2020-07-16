import pandas as pd
from Import_Data import Import_Data
import time
import sklearn.utils

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

start_time = time.time()

array = [['Test', 'Algorithm', 'Bayes_Best_Score', 'Accuracy_Score', 'Bayes_Best_Params']]
array_cross = [['Test', 'Algorithm', 'Cross_Score', 'Accuracy_Score']]

# Import Data
address = 'ML_Data_CV.xlsx'

Training_Data, Training_Database = Import_Data(address, "Training")
Testing_Data, Testing_Database = Import_Data(address, "Testing")

# Start of Loop
index = 0

# Data_Analysis

from Data_Analysis import DA
DA(Training_Data, Testing_Data)


while index <= 1:

    # Shuffle data
    Training_Data = sklearn.utils.shuffle(Training_Data)
    Training_Data.to_csv(str(index + 1) + '_Training_Data_RF.csv')
    Testing_Data = sklearn.utils.shuffle(Testing_Data)
    Testing_Data.to_csv(str(index + 1) + '_Testing_Data_RF.csv')



    #Preliminary Results Cross-Validation
    from Random_Forest_Cross import RF_Cross
    Pred_RF_Cross, Cross_Score_RF_Cross, AccuScore_RF_Cross = RF_Cross(Training_Data, Testing_Data, index)
    array_cross.append([str(index + 1), 'RF', Cross_Score_RF_Cross, AccuScore_RF_Cross])

    from NB_Gaussian_Cross import NB_G_Cross
    Pred_NBG_Cross, Cross_Score_NBG_Cross, AccuScore_NBG_Cross = NB_G_Cross(Training_Data, Testing_Data, index)
    array_cross.append([str(index + 1),'NBG', Cross_Score_NBG_Cross, AccuScore_NBG_Cross])

    from Nearest_Cross import KNB_Cross
    Pred_KNB_Cross, Cross_Score_KNB_Cross, AccuScore_KNB_Cross = KNB_Cross(Training_Data, Testing_Data, index)
    array_cross.append([str(index + 1), 'KNB', Cross_Score_KNB_Cross, AccuScore_KNB_Cross])

    from SVC_Cross import SVC_Cross
    Pred_SVC_Cross, Cross_Score_SVC_Cross, AccuScore_SVC_Cross = SVC_Cross(Training_Data, Testing_Data, index)
    array_cross.append([str(index + 1), 'SVC', Cross_Score_SVC_Cross, AccuScore_SVC_Cross])


# Start of Hyper Parameter Tuning
    print("Test Number " + str(index+1))

# Random Forest
    from Random_Forest import RF
    Pred_RF, Best_Result_RF = RF(Training_Data, Testing_Data, index)
    print("--- %s seconds --- RF Time" % (time.time() - start_time))
    Best_Result_RF.insert(0, "RF")
    Best_Result_RF.insert(0, str(index+1))
    array.append(Best_Result_RF)

# NG_Gaussian
    from NB_Gaussian import NB_G
    Pred_NB_G, Best_Result_NB_G = NB_G(Training_Data, Testing_Data, index)
    print("--- %s seconds --- NG_G Time" % (time.time() - start_time))
    Best_Result_NB_G.insert(0, "NB_G")
    Best_Result_NB_G.insert(0, str(index + 1))
    array.append(Best_Result_NB_G)

# SVC
    from SVC import SVC
    Pred_SVC, Best_Result_SVC = SVC(Training_Data, Testing_Data, index)
    print("--- %s seconds --- SVM Time" % (time.time() - start_time))
    Best_Result_SVC.insert(0, "SVC")
    Best_Result_SVC.insert(0, str(index + 1))
    array.append(Best_Result_SVC)

# KNN
    from Nearest import KNB
    Pred_KNB, Best_Result_KNB = KNB (Training_Data, Testing_Data, index)
    print("--- %s seconds --- KNB Time" % (time.time() - start_time))
    Best_Result_KNB.insert(0, "KNB")
    Best_Result_KNB.insert(0, str(index + 1))
    array.append(Best_Result_KNB)

    print("--- %s seconds --- Total Time" % (time.time() - start_time))
    index = index + 1


# Export best results to Excel

df_cross = pd.DataFrame(array_cross)
df_cross.to_excel('C:/Users/Miguel Jorquera/PycharmProjects/MachineLearning/Cross_Results.xlsx')

df = pd.DataFrame(array)
df.to_excel('C:/Users/Miguel Jorquera/PycharmProjects/MachineLearning/Best_Results.xlsx')

