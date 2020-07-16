from data_preprocess import Preprocess
import torch
from evaluation import get_accuracy_test,get_confusion_matrix,get_micro_f1,get_macro_f1,get_accuracy_ens_bow,get_accuracy_ens_bilstm

class Test:
    def __init__(self,config):
        self.config = config
    
    def test(self):
        prpr = Preprocess(self.config)
        prpr.preprocess(self.config['path_test'],"test")
        labels,sentences,vocabulary,voca_embs,sens_rep,labels_index,labels_rep = prpr.load_preprocessed()
        
        y_real = labels_rep
        lengths = list()
        for sen in sens_rep:
            lengths.append(len(sen))

        if(self.config["model"] == 'bow_ens'):
            models = list()
            for ens in range(int(self.config["ensemble_size"])):
                models.append(torch.load(self.config["path_model"]+'.'+str(ens)))
            x = sens_rep
            accs,y_pre = get_accuracy_ens_bow(models,x,y_real)
            conf_mat = get_confusion_matrix(y_real,y_pre,len(labels_index))
            micro_f1 = get_micro_f1(conf_mat)
            macro_f1,f1 = get_macro_f1(conf_mat)
            output = open(self.config["path_eval_result"],'w')
            print("F1-score of each classes:\n",file=output)
            for i in range(len(labels_index)):
                print(list(labels_index.keys())[i],f1[i],file=output)
            print("{0:<15}\t{1:<15}\t{2}".format("Actual","Prediction","Correct?"),file = output)
            for i,j in zip(y_real,y_pre):
                real = list(labels_index.keys())[list(labels_index.values()).index(i)]
                pre = list(labels_index.keys())[list(labels_index.values()).index(j)]
                if i==j:
                    print("{0:<15}\t{1:<15}\t{2}".format(real,pre,"True"),file = output)
                else:
                    print("{0:<15}\t{1:<15}\t{2}".format(real,pre,"False"),file = output)
            print("The accuracy of {} models are: ".format(len(models)),file=output)
            for i in range(len(accs)-1):
                print(str(accs[i])+" ",file=output)
            print("\nThe ensembled accuracy is: {}\n".format(accs[len(accs)-1]),file=output)
            print(
                "Confusion Matrix:\n",conf_mat,
                "\nMicro F1: ",micro_f1,
                "\nMacro F1: ",macro_f1,
                file = output
            )
            output.close()
            return

        if(self.config["model"] == 'bilstm_ens' or self.config["model"] == 'bow_bilstm_ens'):
            models = list()
            for ens in range(int(self.config["ensemble_size"])):
                models.append(torch.load(self.config["path_model"]+'.'+str(ens)))
            x = torch.nn.utils.rnn.pad_sequence(sens_rep,padding_value=0)
            accs,y_pre = get_accuracy_ens_bilstm(models,x,y_real,lengths)
            conf_mat = get_confusion_matrix(y_real,y_pre,len(labels_index))
            micro_f1 = get_micro_f1(conf_mat)
            macro_f1,f1 = get_macro_f1(conf_mat)
            output = open(self.config["path_eval_result"],'w')
            print("F1-score of each classes:\n",file=output)
            for i in range(len(labels_index)):
                print(list(labels_index.keys())[i],f1[i],file=output)
            print("{0:<15}\t{1:<15}\t{2}".format("Actual","Prediction","Correct?"),file = output)
            for i,j in zip(y_real,y_pre):
                real = list(labels_index.keys())[list(labels_index.values()).index(i)]
                pre = list(labels_index.keys())[list(labels_index.values()).index(j)]
                if i==j:
                    print("{0:<15}\t{1:<15}\t{2}".format(real,pre,"True"),file = output)
                else:
                    print("{0:<15}\t{1:<15}\t{2}".format(real,pre,"False"),file = output)
            print("The accuracy of {} models are: ".format(len(models)),file=output)
            for i in range(len(accs)-1):
                print(str(accs[i])+" ",file=output)
            print("\nThe ensembled accuracy is: {}\n".format(accs[len(accs)-1]),file=output)
            print(
                "Confusion Matrix:\n",conf_mat,
                "\nMicro F1: ",micro_f1,
                "\nMacro F1: ",macro_f1,
                file = output
            )
            output.close()
            return


        model = torch.load(self.config["path_model"])

        if(self.config["model"] == 'bow'):
            x = sens_rep
            acc,y_pre = get_accuracy_test(model,"bow",x,y_real,False)
        if(self.config["model"] == 'bilstm'):
            x = torch.nn.utils.rnn.pad_sequence(sens_rep,padding_value=0)
            acc,y_pre = get_accuracy_test(model,"bilstm",x,y_real,lengths)
        if(self.config["model"] == 'bow_bilstm'):
            x = torch.nn.utils.rnn.pad_sequence(sens_rep,padding_value=0)
            acc,y_pre = get_accuracy_test(model,"bow_bilstm",x,y_real,lengths)

        conf_mat = get_confusion_matrix(y_real,y_pre,len(labels_index))
        micro_f1 = get_micro_f1(conf_mat)
        macro_f1,f1 = get_macro_f1(conf_mat)
        print("Accuracy: {}".format(acc))
        print("Confusion Matrix:\n",conf_mat)
        print("Micro F1: ",micro_f1)
        print("Macro F1: ",macro_f1)
        output = open(self.config["path_eval_result"],'w')
        print("F1-score of each classes:\n",file=output)
        for i in range(len(labels_index)):
            print(list(labels_index.keys())[i],f1[i],file=output)
        print("{0:<15}\t{1:<15}\t{2}".format("Actual","Prediction","Correct?"),file = output)
        for i,j in zip(y_real,y_pre):
            real = list(labels_index.keys())[list(labels_index.values()).index(i)]
            pre = list(labels_index.keys())[list(labels_index.values()).index(j)]
            if i==j:
                print("{0:<15}\t{1:<15}\t{2}".format(real,pre,"True"),file = output)
            else:
                print("{0:<15}\t{1:<15}\t{2}".format(real,pre,"False"),file = output)
        print(
            "The accuray after training is: ",acc,
            '\n',
            "Confusion Matrix:\n",conf_mat,
            '\n',
            "Micro F1: ",micro_f1,
            '\n',
            "Macro F1: ",macro_f1,
            file = output
        )
        output.close()
