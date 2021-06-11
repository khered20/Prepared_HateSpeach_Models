import numpy as np
# from keras import backend as K
import sklearn


class measures:
    
    
    # def recall(y_true, y_pred):
    #     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    #     recall = true_positives / (possible_positives + K.epsilon())
    #     return recall
    
    # def precision(y_true, y_pred):
    #     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    #     precision = true_positives / (predicted_positives + K.epsilon())
    #     return precision
    
    # def f1(y_true, y_pred):
    #     precision1 = measures.precision(y_true, y_pred)
    #     recall1 = measures.recall(y_true, y_pred)
    #     return 2*((precision1*recall1)/(precision1+recall1+K.epsilon()))
    
    def getAcc(y_true, y_pred):
        matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
        TP,TN,FP,FN = matrix[1][1],matrix[0][0],matrix[0][1],matrix[1][0]
        Accuracy = (TP+TN)/(TP+FP+FN+TN)
        return Accuracy
    
    # Evaluation Metrics
    def print_evaluation_metrics(y_true, y_pred, label='', is_regression=True, label2=''):
        print('==================', label2)
        ### For regression
        if is_regression:
            print('mean_absolute_error',label,':', sklearn.metrics.mean_absolute_error(y_true, y_pred))
            print('mean_squared_error',label,':', sklearn.metrics.mean_squared_error(y_true, y_pred))
            print('r2 score',label,':', sklearn.metrics.r2_score(y_true, y_pred))
            #     print('max_error',label,':', sklearn.metrics.max_error(y_true, y_pred))
            return sklearn.metrics.mean_squared_error(y_true, y_pred)
        else:
            ### FOR Classification
            print('balanced_accuracy_score',label,':', sklearn.metrics.balanced_accuracy_score(y_true, y_pred))
            print('average_precision_score',label,':', sklearn.metrics.average_precision_score(y_true, y_pred))
            print('balanced_accuracy_score',label,':', sklearn.metrics.balanced_accuracy_score(y_true, y_pred))
            print('accuracy_score',label,':', sklearn.metrics.accuracy_score(y_true, y_pred))
            print('f1_score',label,':', sklearn.metrics.f1_score(y_true, y_pred))
            
            matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
            print(matrix)
            TP,TN,FP,FN = matrix[1][1],matrix[0][0],matrix[0][1],matrix[1][0]
            Accuracy = (TP+TN)/(TP+FP+FN+TN)
            Precision = TP/(TP+FP)
            Recall = TP/(TP+FN)
            F1 = 2*(Recall * Precision) / (Recall + Precision)
            print('Acc', Accuracy, 'Prec', Precision, 'Rec', Recall, 'F1',F1)
            return sklearn.metrics.accuracy_score(y_true, y_pred)

    

    def getScores(y_true, y_pred):
        matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
        TP,TN,FP,FN = matrix[1][1],matrix[0][0],matrix[0][1],matrix[1][0]
        Accuracy = (TP+TN)/(TP+FP+FN+TN)
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        F1 = 2*(Recall * Precision) / (Recall + Precision)
        
        c2_Precision=TN/(TN+FN)
        c2_Recall = TN/(TN+FP)
        c2_F1 = 2*(c2_Recall * c2_Precision) / (c2_Recall + c2_Precision)
        
        return matrix, Accuracy,F1,Precision,Recall,c2_F1, c2_Precision, c2_Recall
    
    def getMacroAndWeightedScores(y_true, y_pred):
        matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
        TP,TN,FP,FN = matrix[1][1],matrix[0][0],matrix[0][1],matrix[1][0]
        Accuracy = (TP+TN)/(TP+FP+FN+TN)
        
        c1Precision = TP/(TP+FP)
        c1Recall = TP/(TP+FN)
        c1F1 = 2*(c1Recall * c1Precision) / (c1Recall + c1Precision)
        
        c2Precision = TN/(TN+FN)
        c2Recall = TN/(TN+FP)
        c2F1 = 2*(c2Recall * c2Precision) / (c2Recall + c2Precision)
        
        MacroPrecision = (c1Precision+c2Precision)/2
        MacroRecall = (c1Recall+c2Recall)/2
        MacroF1 = (c1F1+c2F1)/2
        
        c1count=np.unique(y_true, return_counts=True)[1][1]
        c2count=np.unique(y_true, return_counts=True)[1][0]
        
        WeightedMacroPrecision = (c1Precision*c1count+c2Precision*c2count)/(c1count+c2count)
        WeightedMacroRecall = (c1Recall*c1count+c2Recall*c2count)/(c1count+c2count)
        WeightedMacroF1 = (c1F1*c1count+c2F1*c2count)/(c1count+c2count)
        
        #from sklearn.metrics import precision_recall_fscore_support
        #precision_recall_fscore_support(tw_labels, y_pred, average='weighted')
        
        from sklearn.metrics import precision_score
        MicroPrecision=precision_score(y_true, y_pred, average='micro')
        
        from sklearn.metrics import recall_score
        MicroRecall=recall_score(y_true, y_pred, average='micro')
        
        from sklearn.metrics import f1_score
        MicroF1=f1_score(y_true, y_pred, average='micro')
        
        return MacroPrecision,MacroRecall,MacroF1,WeightedMacroPrecision,WeightedMacroRecall,WeightedMacroF1,MicroPrecision,MicroRecall,MicroF1
    
    #print_evaluation_metrics([1,0], [0.9,0.1], '', True)
    #print_evaluation_metrics([1,0], [1,1], '', False)
