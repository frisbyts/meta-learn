from pickle import load
import numpy as np 
from scipy.stats import sem 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.backend import clear_session
from read_interaction_matrix import process_data
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from Bio import SubsMat
from Bio import AlignIO
from Bio import Alphabet
from Bio.Alphabet import IUPAC
from Bio.Align import AlignInfo
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#adapted from https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/
# define the model
def define_model(pdz_seq_length,pep_seq_length,num_pdz_aa,num_pep_aa):
    # channel 1
    inputs1 = Input(shape=(pdz_seq_length,num_pdz_aa))
    conv1_1 = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs1)
    conv1_2 = Conv1D(filters=32, kernel_size=5, activation='relu')(inputs1)
    conv1_3 = Conv1D(filters=32, kernel_size=7, activation='relu')(inputs1)
    drop1_1 = Dropout(0.5)(conv1_1)
    drop1_2 = Dropout(0.5)(conv1_2)
    drop1_3 = Dropout(0.5)(conv1_3)
    pool1_1 = MaxPooling1D(pool_size=1)(drop1_1)
    pool1_2 = MaxPooling1D(pool_size=1)(drop1_2)
    pool1_3 = MaxPooling1D(pool_size=1)(drop1_3)
    flat1_1 = Flatten()(pool1_1)
    flat1_2 = Flatten()(pool1_2)
    flat1_3 = Flatten()(pool1_3)
    # channel 2
    inputs2 = Input(shape=(pep_seq_length,num_pep_aa))
    conv2_1 = Conv1D(filters=16, kernel_size=1, activation='relu')(inputs2)
    conv2_2 = Conv1D(filters=16, kernel_size=3, activation='relu')(inputs2)
    conv2_3 = Conv1D(filters=16, kernel_size=5, activation='relu')(inputs2)
    drop2_1 = Dropout(0.5)(conv2_1)
    drop2_2 = Dropout(0.5)(conv2_2)
    drop2_3 = Dropout(0.5)(conv2_3)
    pool2_1 = MaxPooling1D(pool_size=1)(drop2_1)
    pool2_2 = MaxPooling1D(pool_size=1)(drop2_2)
    pool2_3 = MaxPooling1D(pool_size=1)(drop2_3)
    flat2_1 = Flatten()(pool2_1)
    flat2_2 = Flatten()(pool2_2)
    flat2_3 = Flatten()(pool2_3)

    # merge
    merged = concatenate([flat1_1,flat1_2,flat1_3,
                          flat2_1,flat2_2,flat2_3])
    # interpretation
    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='multichannel_protein_kshot.png')
    return model

def plot_line(x,file_loc):
    #x- History object; dic; keys=protein family; values=accuracy
    plt.plot(x["accuracy"],label="Training Acc")
    plt.plot(x["val_accuracy"],label="Testing Acc")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(file_loc+"acc.png")
    plt.clf()
 
    plt.plot(x["loss"],label="Training loss")
    plt.plot(x["val_loss"],label="Testing loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(file_loc+"loss.png")
    plt.clf()
     
if __name__ == '__main__':
    
    n_clusters = 4
    k = 5
    num_epochs = 50
    batch_size = 16 
    num_iters = 3
    mode = "k-shot"
    t = pd.read_csv("t.csv",header=None).values
    p = process_data(t,n_clusters,"mouse_pdz.msa","mouse_peptide.msa")
    m_pdz, m_pep = p.substitution_matrices() 
    X_pdz, X_peptide, y, clusters = p.encode_data()

    if mode == "naive transfer":
        #naive transfer- train on peptide binding on first n-1 families, 
        #                predict peptide binding for nth family.

        for i in range(n_clusters):
            train_indxs = np.where(clusters != i)
            test_indxs = np.where(clusters == i)
            train_pdz = X_pdz[train_indxs]
            train_pep = X_peptide[train_indxs]
            test_pdz = X_pdz[test_indxs]
            test_pep = X_peptide[test_indxs]
            y_train = y[train_indxs] 
            y_test = y[test_indxs]
            pdz_seq_length, num_pdz_aa = train_pdz.shape[1:]
            pep_seq_length, num_pep_aa = train_pep.shape[1:]

            # define model
            model = define_model(pdz_seq_length,pep_seq_length,num_pdz_aa,num_pep_aa)
            # fit model
            history = model.fit([train_pdz,train_pep], y_train, validation_data=([test_pdz,test_pep],y_test),
                                 epochs=num_epochs, batch_size=batch_size)

            plot_line(history.history,"plots/Naive_transfer_{}_".format(i))
            clear_session()

    elif mode == "per family":
        #per family- train separate model of peptide binding for each family,
        #            only make predictions for test cases within that family

        pass

    elif mode == "standard split":
        #standard split- train test split over all data, no separating of families

        train_pdz, test_pdz, y_train, y_test = train_test_split(X_pdz,y,test_size=0.3,random_state=0)
        train_pep, test_pep, _, _ = train_test_split(X_peptide,y,test_size=0.3,random_state=0)
        pdz_seq_length, num_pdz_aa = train_pdz.shape[1:]
        pep_seq_length, num_pep_aa = train_pep.shape[1:]

        model = define_model(pdz_seq_length,pep_seq_length,num_pdz_aa,num_pep_aa)
        history = model.fit([train_pdz,train_pep], y_train, validation_data=([test_pdz,test_pep],y_test), 
                             epochs=num_epochs, batch_size=batch_size)
        
        print(history.history["val_accuracy"][-1])
        plot_line(history.history,"plots/Standard_split_")

    elif mode == "k-shot":
        #k-shot - train over n-1 families, updated model with k positive and negative instances from nth family,
        #         predict peptide binding over the remaining instances from the nth family
        train_accs = []
        k_train_accs = []
        k_only_accs = []
        pan_accs = []
        for j in range(num_iters):
            t_acc = []
            kt_acc = []
            ko_acc = []
            p_acc = []

            #get new shuffles of data
            for d in [X_pdz,X_peptide,y,clusters]:
                np.random.seed(j)
                np.random.shuffle(d)

            for i in range(n_clusters):
                train_indxs = np.where(clusters != i)
                y_train = y[train_indxs] 
                train_pdz = X_pdz[train_indxs]
                train_pep = X_peptide[train_indxs]
                pdz_seq_length, num_pdz_aa = train_pdz.shape[1:]
                pep_seq_length, num_pep_aa = train_pep.shape[1:]

                test_indxs = np.where(clusters == i)
                y_test_all = y[test_indxs]
                X_pdz_test_all = X_pdz[test_indxs]
                X_pep_test_all = X_peptide[test_indxs]

                #get k shot train/test indices
                y_test_positive_indxs = np.where(y_test_all == 1)
                y_test_negative_indxs = np.where(y_test_all == 0)
                k_train_positive_indxs = np.random.choice(y_test_positive_indxs[0],k,replace=False)
                k_train_negative_indxs = np.random.choice(y_test_negative_indxs[0],k,replace=False)
                k_test_positive_indxs = np.setdiff1d(y_test_positive_indxs,k_train_positive_indxs)
                k_test_negative_indxs = np.setdiff1d(y_test_negative_indxs,k_train_negative_indxs)

                #get labels for k shot training
                k_train_positive_labels = y_test_all[k_train_positive_indxs]
                k_train_negative_labels = y_test_all[k_train_negative_indxs]
                k_train_labels = np.hstack((k_train_positive_labels,k_train_negative_labels)) 
                
                #get labels for k shot testing
                k_test_positive_labels = y_test_all[k_test_positive_indxs]
                k_test_negative_labels = y_test_all[k_test_negative_indxs]
                k_test_labels = np.hstack((k_test_positive_labels,k_test_negative_labels))

                #get k-shot test instances
                k_test_positive_pdz = X_pdz[k_test_positive_indxs] 
                k_test_positive_pep = X_peptide[k_test_positive_indxs]
                k_test_negative_pdz = X_pdz[k_test_negative_indxs]
                k_test_negative_pep = X_peptide[k_test_negative_indxs]
                k_test_pdz = np.vstack((k_test_positive_pdz,k_test_negative_pdz))
                k_test_pep = np.vstack((k_test_positive_pep,k_test_negative_pep))

                #get k-shot train instances
                k_train_positive_pdz = X_pdz[k_train_positive_indxs] 
                k_train_positive_pep = X_peptide[k_train_positive_indxs]
                k_train_negative_pdz = X_pdz[k_train_negative_indxs]
                k_train_negative_pep = X_peptide[k_train_negative_indxs]
                k_train_pdz = np.vstack((k_train_positive_pdz,k_train_negative_pdz))
                k_train_pep = np.vstack((k_train_positive_pep,k_train_negative_pep))

                #train without nth family
                model = define_model(pdz_seq_length,pep_seq_length,num_pdz_aa,num_pep_aa)
                meta_phase = model.fit([train_pdz,train_pep], y_train, validation_split=0.2,
                                       epochs=num_epochs, batch_size=batch_size)
                #meta_acc = model.evaluate([k_test_pdz,k_test_pep],k_test_labels)[1]
                meta_preds = model.predict([k_test_pdz,k_test_pep])
                t_acc.append(roc_auc_score(k_test_labels,meta_preds))
     
                #update model with k instances of nth family
                k_shot_update = model.fit([k_train_pdz,k_train_pep],k_train_labels)#,validation_data=([k_test_pdz,k_test_pep],k_test_labels))
                #k_shot_acc = model.evaluate([k_test_pdz,k_test_pep],k_test_labels)[1]
                k_shot_preds = model.predict([k_test_pdz,k_test_pep])
                kt_acc.append(roc_auc_score(k_test_labels,k_shot_preds))

                clear_session()

                #pan family model
                x_train_pdz = np.vstack((train_pdz,k_train_pdz))
                x_train_pep = np.vstack((train_pep,k_train_pep))
                pdz_seq_length, num_pdz_aa = x_train_pdz.shape[1:]
                pep_seq_length, num_pep_aa = x_train_pep.shape[1:]
                y_train = np.hstack((y_train,k_train_labels))

                for d in [x_train_pdz,x_train_pep,y_train]:
                    np.random.seed(5555)
                    np.random.shuffle(d)
                pan_model = define_model(pdz_seq_length,pep_seq_length,num_pdz_aa,num_pep_aa)
                pan_phase = pan_model.fit([x_train_pdz,x_train_pep],y_train,validation_split=0.2,
                                           epochs=num_epochs,batch_size=batch_size)
                #pan_acc = pan_model.evaluate([k_test_pdz,k_test_pep],k_test_labels)[1]
                pan_preds = pan_model.predict([k_test_pdz,k_test_pep])
                p_acc.append(roc_auc_score(k_test_labels,pan_preds))

                clear_session()
     
                #train new model with only k instances of nth family
                k_pdz_length, k_num_pdz = k_train_pdz.shape[1:]
                k_pep_length, k_num_pep = k_train_pep.shape[1:]
                k_shot_only = define_model(k_pdz_length,k_pep_length,k_num_pdz,k_num_pep)
                k_history = k_shot_only.fit([k_train_pdz,k_train_pep],k_train_labels, epochs=num_epochs, batch_size=batch_size)#validation_data=([k_test_pdz,k_test_pep],k_test_labels),
                #k_only_acc = k_shot_only.evaluate([k_test_pdz,k_test_pep],k_test_labels)[1]    
                k_only_preds = k_shot_only.predict([k_test_pdz,k_test_pep])        
                ko_acc.append(roc_auc_score(k_test_labels,k_only_preds))

                clear_session()

            train_accs.append(t_acc)
            k_train_accs.append(kt_acc)
            k_only_accs.append(ko_acc)
            pan_accs.append(p_acc)

        train_accs = np.array(train_accs)
        k_train_accs = np.array(k_train_accs)
        k_only_accs = np.array(k_only_accs)
        pan_accs = np.array(pan_accs)

        train_sem = sem(train_accs,axis=0)
        k_train_sem = sem(k_train_accs,axis=0)
        k_only_sem = sem(k_only_accs,axis=0)
        pan_sem = sem(pan_accs,axis=0)

        print("Meta AUCs:   {}".format(np.mean(train_accs,axis=0)))
        print("k-shot AUCs: {}".format(np.mean(k_train_accs,axis=0)))
        print("k-only AUCs: {}".format(np.mean(k_only_accs,axis=0)))
        print("Pan AUCs:    {}".format(np.mean(pan_accs,axis=0)))

        x = [str(i) for i in range(n_clusters)]
        plt.bar(range(n_clusters),np.mean(train_accs,axis=0),yerr=train_sem,capsize=5,alpha=0.5,label="Meta AUC",color="r",ecolor="r")
        plt.bar(range(n_clusters),np.mean(k_train_accs,axis=0),yerr=k_train_sem,capsize=5,alpha=0.5,label="{}-shot AUC".format(k),color="b",ecolor="b")
        plt.xticks(range(n_clusters),x)
        plt.xlabel("Protein family use for testing")
        plt.ylabel("AUC")
        plt.legend()
        plt.savefig("plots/5-shot_vs_meta_acc_.png")
        plt.show()
        plt.clf()

        plt.bar(range(n_clusters),np.mean(pan_accs,axis=0),yerr=pan_sem,capsize=5,alpha=0.5,label="Pan AUC",color="r",ecolor="r")
        plt.bar(range(n_clusters),np.mean(k_train_accs,axis=0),yerr=k_train_sem,capsize=5,alpha=0.5,label="{}-shot AUC".format(k),color="b",ecolor="b")
        plt.xticks(range(n_clusters),x)
        plt.xlabel("Protein family use for testing")
        plt.ylabel("AUC")
        plt.legend()
        plt.savefig("plots/5-shot_vs_pan_acc.png")
        plt.show()

        plt.bar(range(n_clusters),np.mean(k_only_accs,axis=0),yerr=k_only_sem,capsize=5,alpha=0.5,label="{}-only AUC".format(k),color="r",ecolor="r")
        plt.bar(range(n_clusters),np.mean(k_train_accs,axis=0),yerr=k_train_sem,capsize=5,alpha=0.5,label="{}-shot AUC".format(k),color="b",ecolor="b")
        plt.xticks(range(n_clusters),x)
        plt.xlabel("Protein family use for testing")
        plt.ylabel("AUC")
        plt.legend()
        plt.savefig("plots/5-shot_vs_5-only_acc.png")
        plt.show()





