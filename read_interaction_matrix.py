import pandas as pd 
import numpy as np 
from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from Bio import SubsMat
from Bio import AlignIO
from Bio import Alphabet
from Bio.Alphabet import IUPAC
from Bio.Align import AlignInfo
import matplotlib.pyplot as plt 
import sys

class process_data:
    def __init__(self,interaction_matrix,n_clusters,kinase_msa,peptide_msa):
        self.interaction_matrix = interaction_matrix
        self.n_clusters = n_clusters
        self.kinase_msa = kinase_msa
        self.peptide_msa = peptide_msa

    def _spectral_bicluster(self,n_clusters,interaction_matrix):
        clustering = SpectralBiclustering(n_clusters=n_clusters,random_state=0).fit(self.interaction_matrix)
        pdz_clusters = clustering.row_labels_
        peptide_clusters = clustering.column_labels_
        return pdz_clusters, peptide_clusters 

    def _kmeans(self,n_clusters,interaction_matrix):
        clustering = KMeans(n_clusters=n_clusters,random_state=0).fit(interaction_matrix)
        return clustering.labels_

    def _listify_msas(self):
        msas = []
        for msa in [self.kinase_msa,self.peptide_msa]:
            alignments = []
            with open(msa,"r") as seqs:
                for line in seqs:
                    line = line.strip("\n").split("\t")
                    seq = line[1]
                    alignments.append(seq)
            msas.append(alignments)
        return msas[0], msas[1]

    def substitution_matrices(self):
        subs = []
        for msa in [self.kinase_msa,self.peptide_msa]:
            c_align = AlignIO.read(msa, "tab")
            summary_align = AlignInfo.SummaryInfo(c_align)
            replace_info = summary_align.replacement_dictionary()
            my_arm = SubsMat.SeqMat(replace_info)
            my_lom = SubsMat.make_log_odds_matrix(my_arm)
            subs.append(my_lom)
        return subs[0], subs[1] 

    def _get_encoding(self,seq,sub_matrix):
        encoding_from_sub_matrix = []
        alpha = ["-","A","C","D","E","F","G",
                 "H","I","K","L","M","N","P",
                 "Q","R","S","T","V","W","Y"]
        for aa in seq:
            aa_encoding = []
            for sym in alpha:
                if (aa,sym) in sub_matrix:
                    aa_encoding.append(sub_matrix[(aa,sym)])
                elif (sym,aa) in sub_matrix:
                    aa_encoding.append(sub_matrix[(sym,aa)])
            encoding_from_sub_matrix.append(aa_encoding)
        return encoding_from_sub_matrix

    def encode_data(self):
        pdz_families, _ = self._spectral_bicluster(self.n_clusters,self.interaction_matrix)
        pdz_msas, peptide_msas = self._listify_msas()
        pdz_sub, peptide_sub = self.substitution_matrices()

        init_X_pdz, init_X_peptide = [], []
        for x, msa, sub in zip([init_X_pdz,init_X_peptide],
                               [pdz_msas,peptide_msas],
                               [pdz_sub,peptide_sub]):
            for seq in msa:
                x.append(self._get_encoding(seq,sub))
        X_pdz, X_peptide, y, pdz_family_mapping = [], [], [], []
        for i, pdz in enumerate(init_X_pdz):
            for j, pep in enumerate(init_X_peptide):
                interaction_score = self.interaction_matrix[i,j]
                if interaction_score == 0:
                    continue
                X_pdz.append(pdz)
                X_peptide.append(pep)
                pdz_family_mapping.append(pdz_families[i])
                if interaction_score > 0:
                    y.append(1)
                else:
                    y.append(0)
        return np.array(X_pdz), np.array(X_peptide), np.array(y), np.array(pdz_family_mapping)
        

if __name__ == '__main__':

    t = pd.read_csv("t.csv",header=None).values
    print(t.shape)
    '''
    X = t.values.T
    pca = PCA(2)
    X_pca = pca.fit_transform(X)
    pc1 = X_pca[:,0]
    pc2 = X_pca[:,1]
    plt.scatter(pc1,pc2)
    plt.show()

    '''
    p = process_data(t,4,"mouse_pdz.msa","mouse_peptide.msa")
    m_pdz, m_pep = p.substitution_matrices()
    X_pdz, X_peptide, y, pdz_families = p.encode_data() #try predicting binding affinity

    print(X_pdz.shape,X_peptide.shape,y.shape,pdz_families.shape)
    print(list(pdz_families))

    plt.hist(pdz_families,bins=4)
    plt.xticks(list(range(4)),["0","1","2","3"])
    plt.show()
    

    




