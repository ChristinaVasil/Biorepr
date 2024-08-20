#!/usr/bin/python3
import sys
import numpy as np
import pickle
import argparse
# import pydot
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import graphviz
from os.path import exists
import csv
import copy
import itertools
import multiprocessing as mp
import seaborn as sns
from time import perf_counter
from queue import Queue
from queue import Empty
import networkx as nx
from networkx import write_multiline_adjlist, read_multiline_adjlist
# from networkx.drawing.nx_pydot import pydot_layout
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_pydot import write_dot
import logging
from threading import Thread, Lock
from scipy.stats import pearsonr, wilcoxon

# WARNING: This line is important for 3d plotting. DO NOT REMOVE
from mpl_toolkits.mplot3d import Axes3D

from sklearn import tree
from sklearn import decomposition
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB

#from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay 
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_score, cross_validate, LeaveOneOut
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import tensorflow as tf  
from tensorflow import keras
from keras.models import load_model
from keras.callbacks import ModelCheckpoint




# Prefix for intermediate files
Prefix = ""
THREADS_TO_USE = mp.cpu_count()  # Init to all CPUs
FEATURE_VECTOR_FILENAME = "/home/thlamp/tcga/bladder_results/normalized_data_integrated_matrix.txt"
os.chdir("/home/thlamp/scripts")
lock = Lock()
mpl_lock = Lock()

def progress(s):
    sys.stdout.write("%s" % (str(s)))
    sys.stdout.flush()

# Create a custom logger
logger = logging.getLogger(__name__)

# Set level of logger
logger.setLevel(logging.INFO)

# Create handlers
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(asctime)s - %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
c_handler.setFormatter(c_format)

# Add handlers to the logger
logger.addHandler(c_handler)

def message(s):
    logger.info(s)


def locateTargetField():
    """
    Searches the index of a given field in the feature file.
    """
    # Open file
    # fInput = open("control.csv", "r")
    fInput = open("/datastore/cvas/output.txt", "r")
    # fInput = open("allTogether.csv", "r")

    # Read first line
    sFirstLine = fInput.readline()
    # Search for index of field of interest (e.g. ...death...)
    iCnt = 0
    for sCur in sFirstLine.split():
        # if ("m_" not in sCur):
        #     print sCur
        if "Death" in sCur:
            message("Found field '%s' with index %d" % (sCur, iCnt))
        iCnt += 1

    fInput.close()


def determineCompleteSamples():
    """
    Determine instances that are mapped to all modalities
    """
    ### Determine instances that are mapped to all modalities
    # All samples dict
    dSamples = dict()

    # Open sample sheet file
    # fModalities = open('gdc_sample_sheet.2020-02-19.tsv')
    fModalities = open('')
    # Ignore header
    sLine = fModalities.readline()
    # For every line
    while sLine != "":
        sLine = fModalities.readline()
        if sLine == "":
            break

        # Get sample ID (field #7 out of 8)
        sSampleID = sLine.split("\t")[6].strip()
        # Get data type (4) and remember it
        sDataType = sLine.split("\t")[3].strip()
        # Initialize samples that were not encountered earlier
        if sSampleID not in dSamples.keys():
            dSamples[sSampleID] = dict()
        # Add data type to list
        dSamples[sSampleID][sDataType] = 1
    fModalities.close()

    # For all collected sample IDs
    message("+ Complete instances:")
    for sSampleID, lDataTypes in dSamples.items():
        # Output which one has exactly three data types
        if len(lDataTypes) == 3:
            message(sSampleID)

    message("\n\n- Incomplete instances:")
    for sSampleID, lDataTypes in dSamples.items():
        # Output which one has exactly three data types
        if len(lDataTypes) < 3:
            message(sSampleID)


def PCAOnControl():
    """
    Apply and visualize PCA on control data.
    """

    message("Opening file...")
    mFeatures_noNaNs, vClass, sampleIDs, feat_names, tumor_stage = initializeFeatureMatrices(False, True)
    mFeatures_noNaNs = getControlFeatureMatrix(mFeatures_noNaNs, vClass)
    message("Opening file... Done.")
    X, pca3DRes = getPCA(mFeatures_noNaNs, 3)
    fig = draw3DPCA(X, pca3DRes)
    fig.savefig("controlPCA3D.pdf", bbox_inches='tight')


def PCAOnTumor():
    """
    Apply and visualize PCA on tumor data.
    """
    message("Opening file...")
    mFeatures_noNaNs, vClass, sampleIDs, feat_names, tumor_stage = initializeFeatureMatrices(False, True)
    mFeatures_noNaNs = getNonControlFeatureMatrix(mFeatures_noNaNs, vClass)
    message("Opening file... Done.")
    X, pca3DRes = getPCA(mFeatures_noNaNs, 3)

    fig = draw3DPCA(X, pca3DRes)

    fig.savefig("tumorPCA3D.pdf", bbox_inches='tight')


def draw3DPCA(X, pca3DRes, c=None, cmap=plt.cm.gnuplot, spread=False):
    
    """
    Draw a 3D PCA given, allowing different classes coloring.
    c: This argument allows for different classes to be color-coded in the scatter plot. 
    cmap: The colormap to be used for coloring the data points
    spread:  applies the QuantileTransformer to spread out the data distribution. 
    """
    
    # Percentage of variance explained for each components
    message('explained variance ratio (first 3 components): %s'
            % str(pca3DRes.explained_variance_ratio_))
   
    if spread:
        X = QuantileTransformer(output_distribution='uniform').fit_transform(X)
       
    fig = plt.figure(figsize =(15, 15))
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], edgecolor='k', c=c, cmap=cmap, depthshade=False, s=100)
    ax.set_xlabel("X coordinate (%4.2f)" % (pca3DRes.explained_variance_ratio_[0]), fontsize=20) 
    ax.set_ylabel("Y coordinate (%4.2f)" % (pca3DRes.explained_variance_ratio_[1]), fontsize=20)
    ax.set_zlabel("Z coordinate (%4.2f)" % (pca3DRes.explained_variance_ratio_[2]), fontsize=20)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    fig.show()
    return fig


def getPCA(mFeatures_noNaNs, n_components=3):
    """
    Return the PCA outcome given an array of instances.

    :param mFeatures_noNaNs: The array to analyze.
    :param n_components: The target number of components.
    :return: The PCA transformation result as a matrix.
    """
    pca = decomposition.PCA(n_components)
    pca.fit(mFeatures_noNaNs)
    X = pca.transform(mFeatures_noNaNs)
    return X, pca

def plotExplainedVariance(mFeatures_noNaNs, n_components=100, featSelection = False):
    """
    Save the cumulative plot for the Explained Variance Ratio of PCA.
    :param mFeatures_noNaNs: The array to analyze.
    :param n_components: The target number of components.
    """
    X, pca = getPCA(mFeatures_noNaNs, n_components = n_components)
    cumExplainedVariance = np.cumsum(pca.explained_variance_ratio_)
    pcs=[]
    for pc in range(len(cumExplainedVariance)):
        pcs.append(pc+1)

    PCAdata = { "Variance": cumExplainedVariance, "PCs": pcs}
    plt.clf()
    sns.lineplot(x = 'PCs', y = 'Variance', data = PCAdata, marker="o")
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    if featSelection:
        plt.title('Cumulative Explained Variance Ratio for selected features\n by Principal Components')
    else:
        plt.title('Cumulative Explained Variance Ratio for full vector\n by Principal Components')
    plt.show()
    plt.savefig("cumulative.png")

def rand_jitter(arr):
    """
    Adds a random jitter quantity to an array.

    :param arr: The array to build upon to determine the jitter level.
    :return: The modifier array.
    """
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


# def expander(t):
#     return log10(t)

#def convertTumorType(s):
#    """
#    Converts tumor types to float numbers, based on an index of classes.

#    :param s: The string representing the tumor type.
#    :return: A class index mapped to this type.
#    """
#    fRes = float(["not reported", "stage i", "stage ii", "stage iii", "stage iv", "stage v"].index(s.decode('UTF-8')))
#    if int(fRes) == 0:
#        return np.nan
#    return fRes


def PCAOnAllData(bResetFiles = False):
    """
    Applies and visualizes PCA on all data.
    """
    bResetFiles = False
    if len(sys.argv) > 1:
        if "-resetFiles" in sys.argv: 
            bResetFiles = True

    # Initialize feature matrices
    mFeatures_noNaNs, vClass, sampleIDs, feat_names, tumor_stage = initializeFeatureMatrices(bResetFiles=bResetFiles, bPostProcessing=True)
    message("Applying PCA...")
    X, pca3D = getPCA(mFeatures_noNaNs, 3)

    # Spread
    message("Applying PCA... Done.")

    # Percentage of variance explained for each components
    message('explained variance ratio (first 3 components): %s'
            % str(pca3D.explained_variance_ratio_))

    message('3 components values: %s'
            % str(X))

    message("Plotting PCA graph...")
    # Assign colors
    aCategories, y = np.unique(vClass, return_inverse=True)

    draw3DPCA(X, pca3D, c=y / 2)
    # DEBUG LINES
    message("Returning categories: \n %s" % (str(aCategories)))
    message("Returning categorical vector: \n %s" % (str(y)))
    ####################
    message("Plotting PCA graph... Done.")

def aencoder(x_train, epochs=200):
    """
    Create the autoencoder model.
    :param x_train: the matrix with the training data
    :param epochs: the number of epochs
    :oaram gfeat: variable about the use of graph features or not
    """
    # Define input layer
    encoder_input = keras.Input(shape=(np.shape(x_train)[1], ))
    # Define encoder layers
    encoded  = keras.layers.Dense(2500, activation="relu")(encoder_input)
    encoded  = keras.layers.Dense(500, activation="relu")(encoded)
    encoded  = keras.layers.Dense(100, activation="relu")(encoded)

    # Define encoder layers
    decoded  = keras.layers.Dense(500, activation="relu")(encoded)
    decoded  = keras.layers.Dense(2500, activation="relu")(decoded)
    decoder_output  = keras.layers.Dense(np.shape(x_train)[1], activation="sigmoid")(decoded)

    # Define autoencoder model
    autoencoder = keras.Model(encoder_input, decoder_output)

    autoencoder.summary()

    opt = tf.keras.optimizers.Adam()

    autoencoder.compile(opt, loss='mse')

    # Define ModelCheckpoint callback to save the best model with .keras extension
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Train the autoencoder with ModelCheckpoint callback
    history = autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=24, validation_split=0.10, callbacks=[checkpoint])

def useAencoder(mFeatures):
      # Load the saved model
    loaded_model = load_model('best_model.keras')

    # Create a new model that outputs the encoder part of the loaded model, the 4th layer is the encoder
    encoder_model = keras.Model(inputs=loaded_model.input, outputs=loaded_model.layers[3].output) 

    # Use the encoder model to obtain the compressed representation (100 features) of the input data
    X_encoded = encoder_model.predict(mFeatures)
    return X_encoded

def initializeFeatureMatrices(bResetFiles=False, bPostProcessing=True, bstdevFiltering=False, bNormalize=True, bNormalizeLog2Scale=True, nfeat=50, expSelectedFeats=False, bExportImpMat=False):
    """
    Initializes the case/instance feature matrices, also creating intermediate files for faster startup.

    :param bResetFiles: If True, then reset/recalculate intermediate files. Default: False.
    :param bPostProcessing: If True, then apply post-processing to remove NaNs, etc. Default: True.
    :param bNormalize: If True, then apply normalization to the initial data. Default: True.
    :param bNormalizeLog2Scale: If True, then apply log2 scaling after normalization to the initial data. Default: True.
    :param bstdevFiltering: If True, perform filtering for top variated features per level
    :param nfeat: number of features per level for graphs 
    :param expSelectedFeats: If True, save selected feature names to txt
    :return: The initial feature matrix of the cases/instances.
    """

    message("Opening files...")

    try:
        if bResetFiles:
            raise Exception("User requested file reset...")
        message("Trying to load saved data...")

        # Apply np.load hack
        ###################
        # save np.load
        np_load_old = np.load 

        # modify the default parameters of np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        
        datafile = np.load(Prefix + "patientAndControlData.mat.npy")
        labelfile = np.load(Prefix + "patientAndControlDataLabels.mat.npy")

        # restore np.load for future normal usage
        np.load = np_load_old 
        ####################
        feat_names = getFeatureNames()
        clinicalfile = loadTumorStage()
        message("Trying to load saved data... Done.")
    except Exception as eCur:
        message("Trying to load saved data... Failed:\n%s" % (str(eCur)))
        message("Trying to load saved data from txt...")
        fControl = open(FEATURE_VECTOR_FILENAME, "r")
        message("Loading labels and ids...")
        # labelfile, should have stored tumor_stage or labels?       
        #DEBUG LINES
        message("FILENAME: "+FEATURE_VECTOR_FILENAME)
        ############
        labelfile = np.genfromtxt(fControl, skip_header=1, usecols=(0, 97467),
                                  missing_values=['NA', "na", '-', '--', 'n/a'],
                                  dtype=np.dtype("object"), delimiter=' ').astype(str)
        
        labelfile[:, 0] = np.char.replace(labelfile[:, 0], '"', '')

        fControl.close()
        
        message("This is the label file...")
        message(labelfile)
        
        message("Splitting features, this is the size of labelfile")
        message(np.shape(labelfile))

        message("Loading labels and ids... Done.")
        
        clinicalfile = loadTumorStage()
        
        feat_names = getFeatureNames()

        datafile = loadPatientAndControlData()
        message("Trying to load saved data from txt... Done.")

        # Saving
        saveLoadedData(datafile, labelfile)

    message("Opening files... Done.")
	
    # Split feature set to features/target field
    mFeatures, vClass, sampleIDs, tumor_stage = splitFeatures(clinicalfile, datafile, labelfile)
    
    mControlFeatureMatrix = getControlFeatureMatrix(mFeatures, vClass)
    message("1 .This is the shape of the control matrix:")
    message(np.shape(mControlFeatureMatrix))

    if bExportImpMat:
        exportImputatedMatrix(mFeatures, sampleIDs, feat_names)

    # the new bPostProcessing removes columns from mFeatures and mControlFeatureMatrix
    if bPostProcessing:
        mFeatures, sampleIDs, vClass, feat_names, tumor_stage = postProcessFeatures(mFeatures, vClass, sampleIDs, tumor_stage, feat_names, bstdevFiltering=bstdevFiltering, nfeat=nfeat)
        
    # Update control matrix, taking into account postprocessed data
    mControlFeatureMatrix = getControlFeatureMatrix(mFeatures, vClass)

    message("2 .This is the shape of the control matrix:")
    message(np.shape(mControlFeatureMatrix))

    if bNormalize:
        mFeatures = normalizeData(mFeatures, feat_names, bNormalizeLog2Scale)

    if expSelectedFeats and bstdevFiltering:
        # open file
        with open('exportedSelectedFeatures.txt', 'w+') as f:
            
            # write elements of list
            for items in feat_names:
                f.write('%s\n' %items)
            
        #DEBUG LINES
        message("File written successfully")
        ####################

    # return feat_names in the function with updated postProcessFeatures
    return mFeatures, vClass, sampleIDs, feat_names, tumor_stage


def postProcessFeatures(mFeatures, vClass, sample_ids, tumor_stage, featNames, bstdevFiltering = False, nfeat=50):
    """
    Post-processes feature matrix to replace NaNs with control instance feature mean values, and also to remove
    all-NaN columns.

    :param mFeatures: The matrix to pre-process.
    :param mControlFeatures: The subset of the input matrix that reflects control instances.
    :param sample_ids: A list with sample ids.
    :return: The post-processed matrix, without NaNs.
    """
    message("Replacing NaNs from feature set...")
    # DEBUG LINES
    message("Data shape before replacement: %s" % (str(np.shape(mFeatures))))
    #############

    # WARNING: Imputer also throws away columns it does not like
    # imputer = Imputer(strategy="mean", missing_values="NaN", verbose=1)
    # mFeatures_noNaNs = imputer.fit_transform(mFeatures)

    #rows_to_remove = CheckRowsNaN(mFeatures)
    # DEBUG LINES
    #message("rows_to_remove"+str(sample_ids[rows_to_remove]))
    #    levels_indices = getLevelIndices()
    #############
    
    
    levels_indices = getOmicLevels(featNames)
    # DEBUG LINES
    message("Omic Levels: "+str(levels_indices))
    #############
    #incomplete_samples = incompleteSamples(mFeatures, levels_indices)
    samples_to_remove = incompleteSamples(mFeatures, levels_indices)
    # DEBUG LINES
    message("incomplete_samples"+str(sample_ids[samples_to_remove]))
    #############

    #samples_to_remove = np.concatenate((rows_to_remove, incomplete_samples))
    #samples_to_remove = np.unique(samples_to_remove)

    features_to_remove = CheckColsNaN(mFeatures, levels_indices)
   
    # Remove samples from the matrix
    mFeatures = np.delete(mFeatures, samples_to_remove, axis=0)

    # Remove features from the matrix
    mFeatures = np.delete(mFeatures, features_to_remove, axis=1)

    message("Number of features after filtering: %s" % (str(np.shape(mFeatures))))
    message("Are there any NaNs after filtering?")
    message(np.any(np.isnan(mFeatures[:, :])))

    # Create a boolean mask to keep elements not in the indices_to_remove array
    mask = np.ones(len(sample_ids), dtype=bool)
    mask[samples_to_remove] = False

    message("vClass:"+str(np.shape(vClass)))
    # Use the mask to filter the array
    filtered_sample_ids = sample_ids[mask]
    filtered_vClass = vClass[mask]
    filtered_tumor_stage = tumor_stage[mask]

    features = getFeatureNames()
    
    # Create a new list without the elements at the specified indices
    filtered_features = [element for index, element in enumerate(features) if index not in features_to_remove]
    #DEBUG LINES
    message("filtered_features shape: "+str(np.shape(filtered_features)))
    #############

    message(mFeatures)

    #DEBUG LINES 
    inds = np.where(np.isnan(mFeatures[:, :]))
    print(mFeatures[inds][0:5])
    ############
    
    # imputation for completing missing values using k-Nearest Neighbors
    levels_indices = getOmicLevels(filtered_features)
    
    #DEBUG LINES
    message("levels_indices for methylation" + str(np.shape(levels_indices)))
    ###########

    matrixForKnnImp = mFeatures[:, levels_indices["methylation"][0]:levels_indices["methylation"][1]]

    #DEBUG LINES
    with open("matrixForKnnImp.pickle", "wb") as fOut: 
            pickle.dump(matrixForKnnImp, fOut)
    ###########

    #DEBUG LINES
    message("Matrix shape before transpose: " + str(np.shape(matrixForKnnImp)))
    ###########
    
    matrixForKnnImp = matrixForKnnImp.transpose()

    #DEBUG LINES
    message("Matrix shape after transpose: " + str(np.shape(matrixForKnnImp)))
    ###########
    
    imputer = KNNImputer()
    matrixForKnnImp = imputer.fit_transform(matrixForKnnImp)

    matrixForKnnImp = matrixForKnnImp.transpose()

    #DEBUG LINES
    message("Matrix shape after second transpose: " + str(np.shape(matrixForKnnImp)))
    ###########

    mFeatures[:, levels_indices["methylation"][0]:levels_indices["methylation"][1]] = matrixForKnnImp

    #DEBUG LINES
    with open("afterImputationmFeatures.pickle", "wb") as fOut: 
            pickle.dump(mFeatures, fOut)
    ###########

    #DEBUG LINES 
    print(mFeatures[inds][0:5])
    ############
    # TODO: Check below
    # WARNING: If a control data feature was fully NaN, but the corresponding case data had only SOME NaN,
    # we would NOT successfully deal with the case data NaN, because there would be no mean to replace them by.

    #############
    message("Replacing NaNs from feature set... Done.")

    message("Are there any NaNs after postProcessing?")
    message(np.any(np.isnan(mFeatures[:, :])))

    message("This is mFeatures in postProcessing...")
    message(mFeatures)

    if bstdevFiltering:
        mFeatures, filtered_features = filteringBySD(filtered_features, mFeatures, nfeat=nfeat)
        message("mFeatures shape after stdev filtering: " + str(np.shape(mFeatures)))
        message("filtered_features shape after stdev filtering: " + str(np.shape(filtered_features)))

    return mFeatures, filtered_sample_ids, filtered_vClass, filtered_features, filtered_tumor_stage

def exportImputatedMatrix (mFeatures, sample_ids, feat_names):
    
    levels_indices = getOmicLevels(feat_names)

    matrixForKnnImp = mFeatures[:, levels_indices["methylation"][0]:levels_indices["methylation"][1]]

    #DEBUG LINES
    message("Matrix shape: " + str(np.shape(matrixForKnnImp)))
    message("NaN before removing of empty samples: " + str(np.count_nonzero(np.isnan(matrixForKnnImp))))
    ###########

    # Create a boolean mask where each row is True if it does not contain all NaNs
    mask = np.all(np.isnan(matrixForKnnImp), axis=1)

    # Use the mask to filter out rows with all NaNs
    samples_to_remove = np.where(mask)[0]

    # Remove samples from the matrix
    matrixForKnnImp = np.delete(matrixForKnnImp, samples_to_remove, axis=0)
    
    # Create a boolean mask to keep elements not in the indices_to_remove array
    mask = np.ones(len(sample_ids), dtype=bool)
    mask[samples_to_remove] = False

    # Use the mask to filter the array
    filtered_sample_ids = sample_ids[mask]

    #DEBUG LINES
    message("NaN after removing of empty samples: " + str(np.count_nonzero(np.isnan(matrixForKnnImp))))
    ###########

    levels_indices = {"methylation":levels_indices["methylation"]}

    columns_length = matrixForKnnImp.shape[0]
    
    # Count NaNs per column
    nan_per_column = count_nan_per_column(matrixForKnnImp)
    
    # Compute the frequency of NaNs per column
    nan_frequency = nan_per_column / columns_length
    
    # Initialize a mask for columns to remove based on NaN threshold for all columns
    columns_to_remove = nan_frequency > 0.1
    
    # Get the indices of columns to remove
    columns_to_remove_indices = np.where(columns_to_remove)[0]

    #DEBUG LINES
    message("columns_to_remove_indices: " + str(columns_to_remove_indices))
    message("samples_to_remove: " + str(samples_to_remove))
    ###########

    # Remove features from the matrix
    matrixForKnnImp = np.delete(matrixForKnnImp, columns_to_remove_indices, axis=1)

    features = getFeatureNames()

    features = features[levels_indices["methylation"][0]:levels_indices["methylation"][1]]

    # Create a new list without the elements at the specified indices
    filtered_features = [element for index, element in enumerate(features) if index not in columns_to_remove_indices]

    #DEBUG LINES
    message("Matrix shape before transpose: " + str(np.shape(matrixForKnnImp)))
    ###########
    
    matrixForKnnImp = matrixForKnnImp.transpose()

    #DEBUG LINES
    message("Matrix shape after transpose: " + str(np.shape(matrixForKnnImp)))
    ###########
    
    imputer = KNNImputer()
    matrixForKnnImp = imputer.fit_transform(matrixForKnnImp)

    imputedDf = pd.DataFrame(matrixForKnnImp, index= filtered_features, columns=filtered_sample_ids)

    imputedDf.to_csv('/home/thlamp/tcga/bladder_results/imputedMethylationMatrix.txt')  

def graphVectorPreprocessing(mGraphFeatures):
    """
    :param mGraphFeatures: matrix with the topological features from graph
    :returns the matrix of topological features from graph without the columns that have only one value across all rows
    """

    scaler = StandardScaler()
    scaler.fit(mGraphFeatures)
    mGraphFeatures = scaler.transform(mGraphFeatures)

    # search for columns that have only 0
    res = np.all(mGraphFeatures == 0, axis = 0)
    # keep the indices from the columns except from these with only 0
    resIndex = np.where(~res)[0]
    #DEBUG LINES
    print(resIndex)
    ############
    # remove the columns that have only 0 from the graph matrix
    mGraphFeatures = mGraphFeatures[:, resIndex]
    
    return mGraphFeatures

def getLevelIndices():
    """
    Returns a list with the first and the last columns corresponding to each omic level, by checking the feature ids.
    """
    feature_names = getFeatureNames()

    # Search for elements that start with "ENSG" and contain "."
    indices_of_mrna = np.where(np.core.defchararray.startswith(feature_names, "ENSG") & (np.core.defchararray.find(feature_names, ".") != -1))[0]
    
    # Search for elements that start with "hsa"
    indices_of_mirna = np.where(np.core.defchararray.startswith(feature_names, "hsa"))[0]
    
    # Search for elements that start with "ENSG" and do not contain "."
    indices_of_methylation = np.where(np.core.defchararray.startswith(feature_names, "ENSG") & (np.core.defchararray.find(feature_names, ".") == -1))[0]

    mrna = []
    mirna = []
    methylation = []

    mrna.append(indices_of_mrna[0])
    mrna.append(indices_of_mrna[0] + indices_of_mrna.shape[0])
    message("The columns for the mRNA level are:" + str(mrna))
    mirna.append(indices_of_mirna[0])
    mirna.append(indices_of_mirna[0] + indices_of_mirna.shape[0])
    message("The columns for the miRNA level are:" + str(mirna))
    methylation.append(indices_of_methylation[0])
    methylation.append(indices_of_methylation[0] + indices_of_methylation.shape[0])
    message("The columns for the DNA methylation level are:"+str(methylation))

    all_levels = []
    all_levels.append(mrna)
    all_levels.append(mirna)
    all_levels.append(methylation)
    return all_levels

def getOmicLevels(sfeatureNames):
    """
    :param sfeatureNames: list with the feature names
    :return a list with the first and the last columns corresponding to each omic level, by checking the feature ids.
    """
    # Search for elements that start with "ENSG" and contain "."
    indices_of_mrna = np.where(np.core.defchararray.startswith(sfeatureNames, "ENSG") & (np.core.defchararray.find(sfeatureNames, ".") != -1))[0]

    # Search for elements that start with "hsa"
    indices_of_mirna = np.where(np.core.defchararray.startswith(sfeatureNames, "hsa"))[0]

    # Search for elements that start with "ENSG" and do not contain "."
    indices_of_methylation = np.where(np.core.defchararray.startswith(sfeatureNames, "ENSG") & (np.core.defchararray.find(sfeatureNames, ".") == -1))[0]

    mrna = []
    mirna = []
    methylation = []

    mrna.append(indices_of_mrna[0])
    mrna.append(indices_of_mrna[0] + indices_of_mrna.shape[0])
    mirna.append(indices_of_mirna[0])
    mirna.append(indices_of_mirna[0] + indices_of_mirna.shape[0])
    methylation.append(indices_of_methylation[0])
    methylation.append(indices_of_methylation[0] + indices_of_methylation.shape[0])
    
    omicLevels = {}
    omicLevels["mRNA"] = mrna
    omicLevels["miRNA"] = mirna
    omicLevels["methylation"] = methylation
    #DEBUG LINES
    message("Omic Levels: " + str(omicLevels))
    #########
    return(omicLevels)

def filteringBySD(sfeatureNames, mFeatures, nfeat=50):
    """
    Filters the features by the standard deviation
    :param sfeatureNames: list with the feature names
    :param mFeatures: the feature matrix
    :returns the filtered feature names and the filtered feature matrix 
    """
    omicLevels = getOmicLevels(sfeatureNames)

    filteredIndices = []
    graphFilteredIndices =[]
    for omicLevel, indices in omicLevels.items():
        if omicLevel != "miRNA":
            # calculate standard deviation for each column
            standardDev = np.std(mFeatures[:, indices[0]:indices[1]], axis=0)
          
            # get indices of the top 2000 numbers
            topStandardDev = np.argsort(standardDev)[-2000:]
            graphTopStandardDev = np.argsort(standardDev)[-nfeat:]
            # add in every element of the array the first index of the omic level, in order to keep the full matrix indices
            topStandardDev = topStandardDev + indices[0]
            graphTopStandardDev = graphTopStandardDev + indices[0]
            # add indices to the list
            filteredIndices.extend(topStandardDev.tolist())
            graphFilteredIndices.extend(graphTopStandardDev.tolist())
        else:
            # add indices to the list
            filteredIndices.extend(range(indices[0],indices[1]))

            # calculate standard deviation for each column
            standardDev = np.std(mFeatures[:, indices[0]:indices[1]], axis=0)
            graphTopStandardDev = np.argsort(standardDev)[-nfeat:]
            graphTopStandardDev = graphTopStandardDev + indices[0]
            graphFilteredIndices.extend(graphTopStandardDev.tolist())
    # for omicLevel, indices in omicLevels.items():
    #     if omicLevel != "miRNA":
    #         # calculate standard deviation for each column
    #         standardDev = np.std(mFeatures[:, indices[0]:indices[1]], axis=0)
          
    #         # get indices of the top 2000 numbers
    #         topStandardDev = np.argsort(standardDev)[-2000:]

    #         # add in every element of the array the first index of the omic level, in order to keep the full matrix indices
    #         topStandardDev = topStandardDev + indices[0]

    #         # add indices to the list
    #         filteredIndices.extend(topStandardDev.tolist())
    #     else:
    #         # add indices to the list
    #         filteredIndices.extend(range(indices[0],indices[1]))


    # filter the matrix by the indices
    filteredFeatMatrix = np.take(mFeatures, filteredIndices, axis = 1)
    # filter the features by the indices
    filteredFeats = [sfeatureNames[i] for i in filteredIndices]
    graphFilteredFeats = [sfeatureNames[index] for index in graphFilteredIndices]
    
    # save to csv file
    with open('graphSelectedSDFeats.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(graphFilteredFeats)

    return filteredFeatMatrix, filteredFeats

def CheckRowsNaN(input_matrix, nan_threshold=0.1):
    """
    Returns an array with the index of the rows that were kept after the filtering.
    :param input_matrix: the matrix that will be filtered
    :param nan_threshold: threshold for the frequency of NaN
    """
    message("Rows' filtering... Done")
    
    rows_length = input_matrix.shape[1]
    # count nan per row
    nan_per_row = count_nan_per_row(input_matrix)
    # compute the frequency of nan per row
    nan_frequency  = nan_per_row / rows_length
    # return an array with boolean values, that show the rows with <=nan_threshold 
    rows_to_remove = nan_frequency > nan_threshold

    rows_to_remove = np.where(rows_to_remove)
    # Flatten the 2D array into a 1D array 
    rows_to_remove = np.ravel(rows_to_remove)
    return rows_to_remove

def count_nan_per_row(input_matrix):
    """
    Counts the number of NaNs per row.
    """
    nan_count_per_column = np.sum(np.isnan(input_matrix), axis=1)
    return nan_count_per_column

def incompleteSamples(mAllData, level_indices):
    """
    Returns the indices of the samples that don't have data at all the three omic levels.
    :param mAllData: The full feature matrix of case/instance data.
    :param level_indices: The columns of the omic level to search
    :return: The indices of the rows that don't have data at least in one level
    """
    # create empty array
    indices_of_empty_rows = np.empty(0)
    
    for omic_level, index in level_indices.items():
        # Create a boolean mask indicating NaN values
        nan_mask = np.isnan(mAllData[:, level_indices[omic_level][0]:level_indices[omic_level][1]])
    
        # Use np.all along axis 1 to check if all values in each row are True (indicating NaN)
        rows_with_nan = np.all(nan_mask, axis=1)
    
        # Get the indices of rows with NaN
        indices_of_rows_with_nan = np.where(rows_with_nan)[0]
        
        indices_of_empty_rows = np.append(indices_of_empty_rows, indices_of_rows_with_nan)

    indices_of_empty_rows = np.unique(indices_of_empty_rows).astype(int)
    return indices_of_empty_rows

# def CheckColsNaN(input_matrix, levels, nan_threshold=0.2):
#     """
#     Returns an array with the index of the columns that were kept
#     :param input_matrix: the matrix that will be filtered
#     :param nan_threshold: threshold for the frequency of NaN
#     """
#     message("Columns' filtering... ")
#     columns_length = input_matrix.shape[0]

#     # count nan per column
#     nan_per_column = count_nan_per_column(input_matrix)
#     # compute the frequency of nan per column
#     nan_frequency  = nan_per_column / columns_length
    
#     # Count zeros per column
#     zero_per_column = count_zero_per_column(input_matrix[:, levels["mRNA"][0]:levels["miRNA"][1]])
#     # Compute the frequency of zeros per column
#     zero_frequency = zero_per_column / columns_length
    
#     # Identify columns to remove based on NaN and zero frequency thresholds
#     columns_to_remove = (nan_frequency > nan_threshold) | (zero_frequency > nan_threshold)

#     # Get the indices of columns to remove
#     columns_to_remove = np.where(columns_to_remove)[0]

#     # # return an array with boolean values, that show the columns with <=nan_threshold t
#     # columns_to_remove = nan_frequency > nan_threshold

#     # columns_to_remove = np.where(columns_to_remove)
#     # # Flatten the 2D array into a 1D array 
#     # columns_to_remove = np.ravel(columns_to_remove)
#     return columns_to_remove

def CheckColsNaN(input_matrix, levels, nan_threshold=0.1, zero_threshold=0.2):
    """
    Returns an array with the index of the columns that should be removed
    :param input_matrix: the matrix that will be filtered
    :param nan_threshold: threshold for the frequency of NaN
    :param zero_threshold: threshold for the frequency of zeros
    """
    message("Columns' filtering... ")

    columns_length = input_matrix.shape[0]
    #DEBUG LINES
    message("columns_length: "+str(columns_length))
    message("shape: "+str(np.shape(input_matrix)))
    ##############
    # Count NaNs per column
    nan_per_column = count_nan_per_column(input_matrix)
    # Compute the frequency of NaNs per column
    nan_frequency = nan_per_column / columns_length
    
    # Count zeros per column only in mRNA and miRNA
    zero_per_column = count_zero_per_column(input_matrix[:, levels["mRNA"][0]:levels["miRNA"][1]])
    
    # Compute the frequency of zeros per column for mRNA and miRNA
    zero_frequency = zero_per_column / columns_length
    
    # Initialize a mask for columns to remove based on NaN threshold for all columns
    columns_to_remove = nan_frequency > nan_threshold
    
    # Update the mask for the first two columns based on zero threshold
    columns_to_remove[:levels["miRNA"][1]] = columns_to_remove[:levels["miRNA"][1]] | (zero_frequency > zero_threshold)
    
    # Get the indices of columns to remove
    columns_to_remove_indices = np.where(columns_to_remove)[0]

    return columns_to_remove_indices

def count_nan_per_column(input_matrix):
    """
    Counts the number of NaNs per column.
    """
    nan_count_per_column = np.sum(np.isnan(input_matrix), axis=0)
    return nan_count_per_column

def count_zero_per_column(matrix):
    return np.sum(matrix == 0, axis=0)

# TODO add sampleid in splitFeatures

def splitFeatures(clinicalfile, datafile, labelfile): 
    """
    Extracts class and instance info, returning them as separate matrices, where rows correspond to the same
    case/instance.

    :param clinicalfile: The file with the clinical info.
    :param datafile: The matrix containing the full feature data from the corresponding file.
    :param labelfile: The matrix containing  the full label data from the corresponding file.
    :return: A tuple of the form (matrix of features, matrix of labels)
    Chris update: :return: A tuple of the form (matrix of features, matrix of labels, sample ids)
    """
    message("Splitting features...")
    message("Number of features: %d"%(np.size(datafile, 1)))
    message("This is the label file:")
    message(labelfile)
    message("This is the shape of the labelfile: %s" % (str(np.shape(labelfile))))
    mFeatures = datafile[:, :]
    
    # DEBUG LINES
    message("Label file rows: %d\tFeature file rows: %d"%(np.shape(labelfile)[0], np.shape(mFeatures)[0]))
    #############

    tumor_stage = clinicalfile[:, 1]
    
    vClass = labelfile[:, 1]
    sampleIDs = labelfile[:, 0]
    print("This is the vClass: ")
    print(vClass)
    # DEBUG LINES
    message("Found classes:\n%s" % (str(vClass)))
    message("Found sample IDs:\n%s" % (str(sampleIDs)))
    #############

    message("Splitfeatures: This is the mFeatures...")
    message(mFeatures)
    message("Splitting features... Done.")

    return mFeatures, vClass, sampleIDs, tumor_stage


def saveLoadedData(datafile, labelfile):
    """
    Saves intermediate data and label file matrices for quick loading.
    :param datafile: The matrix containing the feature data.
    :param labelfile: The matrix containing the label data.
    """
    message("Saving data in dir..." + os.getcwd())
    np.save(Prefix + "patientAndControlData.mat.npy", datafile)
    np.save(Prefix + "patientAndControlDataLabels.mat.npy", labelfile)
    # np_datafile =  np.array(datafile)
    # np_labelfile = np.array(labelfile)
    # np.savetxt("firstRepresentationDataFile.txt", np_datafile)
    # np.savetxt("firstRepresentationLabelFile.txt", np_labelfile)
    message("Saving data... Done.")


def loadPatientAndControlData():
    """
    Loads and returns the serialized patient and control feature data file as a matrix.
    :return: the patient and control feature data file as a matrix
    """
    message("Loading features...")
    fControl = open(FEATURE_VECTOR_FILENAME, "r")
    datafile = np.genfromtxt(fControl, skip_header=1, usecols=range(1, 97467),
                             missing_values=['NA', "na", '-', '--', 'n/a'], delimiter=" ",
                             dtype=np.dtype("float")
                             )
    fControl.close()

    message("This is the datafile...")
    message(datafile)
    message("Loading features... Done.")
    return datafile


def loadTumorStage():
    """
    Gets tumor stage data from clinical data file.
    :return: A matrix indicating the tumor stage per case/instance.
    """
    message("Loading tumor stage...")
    fControl = open(FEATURE_VECTOR_FILENAME, "r")
    
    clinicalfile = np.genfromtxt(fControl, skip_header=1, usecols=(0, 97468),
                                  missing_values=['NA', "na", '-', '--', 'n/a'],
                                  dtype=np.dtype("object"), delimiter=' ').astype(str)
    
    clinicalfile[:, 0] = np.char.replace(clinicalfile[:, 0], '"', '')
    clinicalfile[:, 1] = np.char.replace(clinicalfile[:, 1], 'NA', '0')
    fControl.close()
    message("Loading tumor stage... Done.")
    message("This is the clinical file...")
    message(clinicalfile)
    message("These are the dimensions of the clinical file")
    message(np.shape(clinicalfile))
    return clinicalfile

def filterTumorStage(mFeatures, vTumorStage, vClass, sampleIDs, mgraphsFeatures=None, useGraphFeatures=False):    
    """
    Filters out the samples that don't have data at tumor stage (tumor stage == 0) and control samples (class = 2) from the feature matrix, graph 
    feature matrix and tumor stage array and returns these objects.
    :param mFeatures: the feature matrix
    :param mgraphsFeatures: the graph feature matrix
    :param vTumorStage: array with tumor stage data
    :param useGraphFeatures: filter also graph features
    """
    # DEBUG LINES
    message(np.shape(mFeatures))
    message(np.shape(vTumorStage))
    ###################
    
    izerosIndex = np.where(vTumorStage == "0")[0]
    iNonControlIndex = np.where(vClass == "2")[0]
    combinedIndex = np.unique(np.concatenate((iNonControlIndex, izerosIndex), axis=None))
    mSelectedFeatures = np.delete(mFeatures, combinedIndex, 0)
    if useGraphFeatures:
        mSelectedGraphFeatures = np.delete(mgraphsFeatures, combinedIndex, 0)
    sselectedTumorStage = np.delete(vTumorStage, combinedIndex, 0)
    selectedvClass = np.delete(vClass, combinedIndex, 0)

    # DEBUG LINES
    message("Zero indices")
    message(izerosIndex)
    message("Non control index")
    message(iNonControlIndex)
    message("combinedIndex")
    message(combinedIndex)
    message("Shape of matrix:")
    message(np.shape(mSelectedFeatures))
    if useGraphFeatures:
        message("Shape of graph matrix:")
        message(np.shape(mSelectedGraphFeatures))
    message("Shape of tumor stage:")
    message(np.shape(sselectedTumorStage))
    ###################
    if useGraphFeatures:
        return mSelectedFeatures, mSelectedGraphFeatures, sselectedTumorStage, selectedvClass
    else:
        return mSelectedFeatures, sselectedTumorStage, selectedvClass

def kneighbors(X, y, lmetricResults, sfeatClass, savedResults):
    """
    Calculates and outputs the performance of classification, through Leave-One-Out cross-valuation, given a set of feature vectors and a set of labels.
    :param X: The feature vector matrix.
    :param y: The labels.
    :param lmetricResults: list for the results of performance metrics.
    :param sfeatClass: string/information about the ML model, the features and data labels 
    :param savedResults: dictionary for the F1-macro results for wilcoxon test
    """
    neigh = KNeighborsClassifier(n_neighbors=1)
    
    # scoring = {
    # 'accuracy': make_scorer(accuracy_score),
    # 'f1_micro': make_scorer(f1_score, average="micro"),
    # 'f1_macro': make_scorer(f1_score, average="macro")}
    
    cv = StratifiedKFold(n_splits=10)
    crossValidation(X, y, cv, neigh, lmetricResults, sfeatClass, savedResults)
    # Calculate cross-validation scores for both accuracy and F1
    # scores = cross_validate(neigh, X, y, cv=cv, scoring=scoring)
    
    # Calculate SEM 
    # sem_accuracy = np.std(scores['test_accuracy']) / np.sqrt(len(scores['test_accuracy']))
    # sem_f1_micro = np.std(scores['test_f1_micro']) / np.sqrt(len(scores['test_f1_micro']))
    # sem_f1_macro = np.std(scores['test_f1_macro']) / np.sqrt(len(scores['test_f1_macro']))

    # message("Avg. Performanace: %4.2f (st. dev. %4.2f, sem %4.2f) \n %s" % (np.mean(scores['test_accuracy']), np.std(scores['test_accuracy']), sem_accuracy, str(scores['test_accuracy'])))
    # message("Avg. F1-micro: %4.2f (st. dev. %4.2f, sem %4.2f) \n %s" % (np.mean(scores['test_f1_micro']), np.std(scores['test_f1_micro']), sem_f1_micro, str(scores['test_f1_micro'])))
    # message("Avg. F1-macro: %4.2f (st. dev. %4.2f, sem %4.2f) \n %s" % (np.mean(scores['test_f1_macro']), np.std(scores['test_f1_macro']), sem_f1_macro, str(scores['test_f1_macro'])))
    
    # lmetricResults.append([sfeatClass, np.mean(scores['test_accuracy']), sem_accuracy, np.mean(scores['test_f1_micro']), sem_f1_micro, 
    #                   np.mean(scores['test_f1_macro']), sem_f1_macro])

def plotAccuracy(df):
    """
    Save the plot with the accuracies from machine learning algorithms with the standard error.
    :param df: the dataframe with the results of the metrics 
    """
    # Plot
    plt.clf()
    sns.barplot(x='Method', y='Mean_Accuracy', data=df, hue='Method', errorbar='se')  
    plt.errorbar(x=df['Method'], y=df['Mean_Accuracy'], yerr=df['SEM_Accuracy'], fmt='o', color='red', capsize=6, elinewidth=3)
    addlabels(df['Mean_Accuracy'], df['SEM_Accuracy'])
    plt.xlabel('Method', fontsize = 20)
    plt.ylabel('Mean Accuracy', fontsize = 20)
    plt.title('Mean Accuracy with Standard Error', fontsize = 25)
    plt.xticks(rotation=45, fontsize = 15)  # Rotate x-axis labels for better readability
    plt.yticks(fontsize = 15)
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig('barplotAccuracy.png')
    plt.show()

def plotF1micro(df):
    """
    Save the plot with the F1 micro from machine learning algorithms with the standard error.
    :param df: the dataframe with the results of the metrics 
    """
    plt.clf()
    sns.barplot(x='Method', y='Mean_F1_micro', data=df, hue='Method', errorbar='se')  
    plt.errorbar(x=df['Method'], y=df['Mean_F1_micro'], yerr=df['SEM_F1_micro'], fmt='o', color='red', capsize=6, elinewidth=3)
    addlabels(df['Mean_F1_micro'], df['SEM_F1_micro'])
    plt.xlabel('Method', fontsize = 20)
    plt.ylabel('Mean F1_micro', fontsize = 20)
    plt.title('Mean F1_micro with Standard Error', fontsize = 25)
    plt.xticks(rotation=45, fontsize = 15)  # Rotate x-axis labels for better readability
    plt.yticks(fontsize = 15)
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig('barplotF1micro.png')
    plt.show()

def plotF1macro(df):
    """
    Save the plot with the F1 macro from machine learning algorithms with the standard error.
    :param df: the dataframe with the results of the metrics 
    """
    plt.clf()
    sns.barplot(x='Method', y='Mean_F1_macro', data=df, hue='Method', errorbar='se')  
    plt.errorbar(x=df['Method'], y=df['Mean_F1_macro'], yerr=df['SEM_F1_macro'], fmt='o', color='red', capsize=6, elinewidth=3)
    addlabels(df['Mean_F1_macro'], df['SEM_F1_macro'])
    plt.xlabel('Method', fontsize = 20)
    plt.ylabel('Mean F1_macro', fontsize = 20)
    plt.title('Mean F1_macro with Standard Error', fontsize = 25)
    plt.xticks(rotation=45, fontsize = 15)  # Rotate x-axis labels for better readability
    plt.yticks(fontsize = 15)
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig('barplotF1macro.png')
    plt.show()

def addlabels(values,stdErr):
    """
    Adds the values of the metrics in the middle of each bar.
    :param values: the values of the metrics
    :param stdErr: the standard error of the values
    """
    for i in range(len(values)):
        label=str(round(values[i], 2))+"("+ str(round(stdErr[i],2))+")"
        plt.text(i, values[i]/2, label, ha = 'center', fontsize=15)


# Find only the control samples
def getControlFeatureMatrix(mAllData, vLabels):
    """
    Gets the features of control samples only.
    :param mAllData: The full matrix of data (control plus tumor data).
    :param vLabels: The matrix of labels per case/instance.
    :return: The subset of the data matrix, reflecting only control cases/instances.
    """
    message("Finding only the control data...")
    choicelist = mAllData
    
    # 0 is the label for controls
    condlist = vLabels == "2"
    message("This is the control feature matrix:")
    print(choicelist[condlist])
    message("Data shape: %s" % (str(np.shape(choicelist[condlist]))))
    message("Finding only the control data...Done")
    return choicelist[condlist]


#def isEqualToString(npaVector, sString):
#    """
#    Compares the string value of a vector to a given string, token by token.
#    :param npaVector: The input vector.
#    :param sString: The string to compare to.
#    :return: True if equal. Otherwise, False.
#    """

#    #TODO check whether we have to convert to UTF-8
#    aRes = np.array([oCur.decode('UTF-8').strip() for oCur in npaVector[:]])
#    aRes = np.array([oCur.strip() for oCur in aRes[:]])
#    aStr = np.array([sString.strip() for oCur in npaVector[:]])
#    return aRes == aStr


def getNonControlFeatureMatrix(mAllData, vLabels):
    """
    Returns the subset of the feature matrix, corresponding to non-control (i.e. tumor) data.
    :param mAllData: The full feature matrix of case/instance data.
    :param vLabels: The label matrix, defining what instance is what type (control/tumor).
    :return: The subset of the feature matrix, corresponding to non-control (i.e. tumor) data
    """
    choicelist = mAllData
    condlist = vLabels == "1"
    message("This is the non control feature matrix:")
    print(choicelist[condlist])
    message("Data shape: %s" % (str(np.shape(choicelist[condlist]))))
    message("Finding only the non control data...Done")
    return choicelist[condlist]


def normalizeData(mFeaturesToNormalize, sfeatureNames, logScale=True):
    """
    Calculates relative change per feature, transforming also to a log 2 norm/scale

    :param mFeaturesToNormalize: The matrix of features to normalize.
    :param sfeatureNames: The names of the columns/features
    :param logScale: If True, log scaling will occur to the result. Default: True.
    :return: The normalized and - possibly - log scaled version of the input feature matrix.
    """
    # DEBUG LINES
    message("Data shape before normalization: %s" % (str(np.shape(mFeaturesToNormalize))))
    #############
    message("Normalizing data...")
    levels = getOmicLevels(sfeatureNames)
    
    # if logScale:
    #     mFeaturesToNormalize[:, levels["mRNA"][0]:levels["miRNA"][1]] = np.log2(2.0 + mFeaturesToNormalize[:, levels["mRNA"][0]:levels["miRNA"][1]])  # Ascertain positive numbers
    
    scaler = MinMaxScaler()
    scaler.fit(mFeaturesToNormalize[:, levels["mRNA"][0]:levels["miRNA"][1]])
    mFeaturesToNormalize[:, levels["mRNA"][0]:levels["miRNA"][1]] = scaler.transform(mFeaturesToNormalize[:, levels["mRNA"][0]:levels["miRNA"][1]])
    # DEBUG LINES
    message("Data shape after normalization: %s" % (str(np.shape(mFeaturesToNormalize))))
    #############
    message("Normalizing based on control set... Done.")
    return mFeaturesToNormalize

def plotDistributions(mFeatures, sfeatureNames, stdfeat, preprocessing):
    """
    Plots the distributions of the values for the three omic levels.
    :param mFeatures: the feature matrix
    :param sfeatureNames: selected feature names
    :param stdfeat: feature selection by standard deviation 
    """
    #levels_indices = getLevelIndices()
    levels_indices = getOmicLevels(sfeatureNames)
    
    for omicLevel, _ in levels_indices.items():

        values_to_plot = mFeatures[:, levels_indices[omicLevel][0]:levels_indices[omicLevel][1]].flatten()
        # Create a mask to identify non-NaN values
        mask = ~np.isnan(values_to_plot)
        # DEBUG LINE
        print("length before nan removing: "+str(len(values_to_plot)))
        ##########################
        # Retrieve only the numbers
        values_to_plot = values_to_plot[mask]
        
        # DEBUG LINE
        print("length after nan removing: "+str(len(values_to_plot)))
        ##########################
        plt.clf()
        fig = plt.figure(figsize=(12, 6))
        plt.hist(values_to_plot,histtype = 'bar', bins = 70)
            
        # x-axis label
        plt.xlabel('Values')
        # frequency label
        plt.ylabel('Counts')
        if stdfeat:
            title ="Data distribution of " + omicLevel + " for feature selection after data preprocessing"
        elif preprocessing: 
            title ="Data distribution of " + omicLevel + " for full vector after data preprocessing"
        else:       
            title ="Data distribution of " + omicLevel + " for full vector"
        # plot title
        plt.title(title)

        # use savefig() before show().
        plt.savefig(omicLevel + "_distribution.png") 

        # function to show the plot
        plt.show()

def plotSDdistributions(mFeatures, sfeatureNames):
    """
    Plots the distributions of the values for the three omic levels.
    :param mFeatures: the feature matrix
    :param sfeatureNames: selected feature names
    """
    #levels_indices = getLevelIndices()
    levels_indices = getOmicLevels(sfeatureNames)
    
    faStdev = np.std(mFeatures, axis=0)
    faStdev = np.log2(1+faStdev)
    for omicLevel, _ in levels_indices.items():
        #DEBUG LINES
        message("Omic Level: " + omicLevel)
        ###########
        values_to_plot = faStdev[levels_indices[omicLevel][0]:levels_indices[omicLevel][1]]
        
        # DEBUG LINE
        message("Length of values for plot: " + str(len(values_to_plot)))
        message("Is there any NaN?")
        message(np.unique(np.isnan(values_to_plot)))
        ##########################
        plt.clf()
        fig = plt.figure(figsize=(12, 6))
        plt.hist(values_to_plot,histtype = 'bar', bins=50)
            
        # x-axis label
        plt.xlabel('log2(Values+1)')
        #plt.xlabel('Values')
        # frequency label
        plt.ylabel('Counts')
        # plot title
        plt.title("Data distribution of standard deviation from " + omicLevel)

        # use savefig() before show().
        plt.savefig(omicLevel + "_SDdistribution.png") 

        # function to show the plot
        plt.show()

def testSpreadingActivation():
    """
    A harmless test of graph drawing and spreading activation effect.
    """
    g = nx.Graph()  

    # adds edges to the graph
    g.add_edge(1, 2, weight=0.5)
    g.add_edge(2, 3, weight=0.5)
    g.add_edge(3, 4, weight=0.5)
    g.add_edge(2, 6, weight=0.2)
    g.add_edge(5, 6, weight=0.8)

    for nNode in g.nodes():
        g.nodes[nNode]['weight'] = nNode * 10
        
    drawAndSaveGraph(g, bSave = False)

    spreadingActivation(g)
    drawAndSaveGraph(g, bSave = False)
    


def getFeatureNames():
    """
    :return: The list of feature names
    """
    message("Loading feature names...")
    # Read the first line from the file
    with open(FEATURE_VECTOR_FILENAME, 'r') as file:
        first_line = file.readline()
    
    # Separate the contents by space and store them in a list
    column_names = first_line.split()
    
    #Remove label and tumor stage
    column_names = column_names[:-2]
    
    # Remove double quotes from all elements in the list
    column_names = [element.replace('"', '') for element in column_names]
    message("Loading feature names... Done.")
    return column_names



def addEdgeAboveThreshold(i, qQueue):
    """
    Helper function for parallel execution. It adds an edge between two features in the overall feature correlation
    graph, if the correlation exceeds a given level. All parameters are provided via a task Queue.
    :param i: The number of the executing thread.
    :param qQueue: The Queue object containing related task info.
    """
    while True:
        # Get next feature index pair to handle
        params = qQueue.get()
        # If empty, stop
        if params is None:
            message("Reached and of queue... Stopping.")
            break
        
        iFirstFeatIdx, iSecondFeatIdx, g, mAllData, saFeatures, iFirstFeatIdx, iSecondFeatIdx, iCnt, iAllPairs, dStartTime, dEdgeThreshold = params

        # DEBUG LINES
        
        if iCnt != 0 and (iCnt % 1000 == 0):
            progress(".")
            if iCnt % 10000 == 0 and iCnt != 0:
                dNow = perf_counter()
                dRate = ((dNow - dStartTime) / iCnt)
                dRemaining = (iAllPairs - iCnt) * dRate
                message("%d (Estimated remaining (sec): %4.2f - Working at a rate of %4.2f pairs/sec)\n" % (
                    iCnt, dRemaining, 1.0 / dRate))

        iCnt += 1
        #############
        
        
        # Fetch feature columns and calculate pearson
        vFirstRepr = mAllData[:, iFirstFeatIdx]
        vSecondRepr = mAllData[:, iSecondFeatIdx]
        fCurCorr = pearsonr(vFirstRepr, vSecondRepr)[0]
        
        # Add edge, if above threshold
        if fCurCorr > dEdgeThreshold:
            g.add_edge(saFeatures[iFirstFeatIdx], saFeatures[iSecondFeatIdx], weight=round(fCurCorr * 100) / 100)## dtrogg se 2 dekadika psifia

        qQueue.task_done()


# Is this the step where we make the generalised graph? The output is one Graph?
def getFeatureGraph(mAllData, saFeatures, dEdgeThreshold=0.30, nfeat=50, bResetGraph=True, stdevFeatSelection=True):
    """
    Returns the overall feature graph, indicating interconnections between features.

    :param mAllData: The matrix containing all case/instance data.
    :param dEdgeThreshold: The threshold of minimum correlation required to keep an edge.
    :param nfeat: number of features per level
    :param bResetGraph: If True, recompute correlations, else load from disc (if available). Default: True.
    :param dMinDivergenceToKeep: The threshold of deviation, indicating which features it makes sense to keep.
    Features with a deviation below this value are considered trivial. Default: log2(10e5).
    :return: The graph containing only useful features and their connections, indicating correlation.
    """

    try:
        if bResetGraph:
            raise Exception("User requested graph recreation.")

        message("Trying to load graph...")
        g = nx.Graph()
        if stdevFeatSelection:
            g = read_multiline_adjlist(Prefix + "graphSDAdjacencyList.txt", create_using=g) ## reads the graph from a file using read_multiline_adjlist
            with open(Prefix + "usefulSDFeatureNames.pickle", "rb") as fIn: ## reads a list of useful feature names from a pickle file
                saUsefulFeatureNames = pickle.load(fIn)
        else:    
            g = read_multiline_adjlist(Prefix + "graphAdjacencyList.txt", create_using=g) ## reads the graph from a file using read_multiline_adjlist
            with open(Prefix + "usefulFeatureNames.pickle", "rb") as fIn: ## reads a list of useful feature names from a pickle file
                saUsefulFeatureNames = pickle.load(fIn)
        message("Trying to load graph... Done.")
        return g, saUsefulFeatureNames
    except Exception as e:
        message("Trying to load graph... Failed:\n%s\n Recomputing..." % (str(e)))

    # DEBUG LINES
    message("Got data of size %s." % (str(np.shape(mAllData))))
    message("Extracting graph...")
    #############
    # Init graph

    # Determine meaningful features (with a divergence of more than MIN_DIVERGENCE from the control mean)
    
    #!iFeatureCount = np.shape(mAllData)[1] ## the number of features in the input data mAllData
    #!mMeans = np.nanmean(mAllData, 0)  # Ignore nans ##computes the mean of each feature, ignoring NaN values.

    #! DEBUG LINES
    #!message("Means: %s"%(str(mMeans)))
    #!dMeanDescribe = pd.DataFrame(mMeans)
    #!print(str(dMeanDescribe.describe()))
    #############
    if stdevFeatSelection:
        fUsefulFeatureNames = open("/home/thlamp/scripts/graphSelectedSDFeats.csv", "r")

        # labelfile, should have stored tumor_stage or labels?       

        saUsefulFeatureNames = np.genfromtxt(fUsefulFeatureNames,
                                missing_values=['NA', "na", '-', '--', 'n/a'],
                                dtype=np.dtype("object"), delimiter=',').astype(str)

    else:
        fUsefulFeatureNames = open("/home/thlamp/tcga/bladder_results/DEGs" +str(nfeat) + ".csv", "r")

        # labelfile, should have stored tumor_stage or labels?       

        saUsefulFeatureNames = np.genfromtxt(fUsefulFeatureNames, skip_header=1, usecols=(0),
                                        missing_values=['NA', "na", '-', '--', 'n/a'],
                                        dtype=np.dtype("object"), delimiter=',').astype(str)
        ##numpy.genfromtxt function to read data from a file. This function is commonly used to load data from text files into a NumPy array.
        ##dtype=np.dtype("object"): This sets the data type for the resulting NumPy array to "object," which is a generic data type that can hold any type of data

        #+ removes " from first column 
        saUsefulFeatureNames[:] = np.char.replace(saUsefulFeatureNames[:], '"', '')

    fUsefulFeatureNames.close()
    # Q1 Chris: is this the step where we apply the threshold? What is the threshold?
    # So, basically keep in vUseful, only the features that their value is greater than dMinDivergenceToKeep
    #!vUseful = [abs(mMeans[iFieldNum]) > dMinDivergenceToKeep for iFieldNum in range(0, iFeatureCount)] ##boolean list indicating whether each feature's absolute deviation from the mean is greater than dMinDivergenceToKeep
    # saFeatures = getFeatureNames()[1:iFeatureCount] ## obtaining the names of the features in the dataset
    # REMOVED and take as input filtered features names from initializeFeatureMatrices

    #!saUsefulIndices = [iFieldNum for iFieldNum, _ in enumerate(saFeatures) if vUseful[iFieldNum]]

    saUsefulIndices = [saFeatures.index(iFieldNum) for iFieldNum in saUsefulFeatureNames if iFieldNum in saFeatures]

    iUsefulFeatureCount = len(saUsefulIndices)
    message("Keeping %d features out of %d." % (len(saUsefulIndices), len(saFeatures)))
    ###############################
    
    g = nx.Graph()
    message("Adding nodes...")
    # Add a node for each feature
    lIndexedNames = enumerate(saFeatures)
    for idx in saUsefulIndices:
        # Only act on useful features
        g.add_node(saFeatures[idx], label=idx)
    message("Adding nodes... Done.")

    # Measure correlations
    iAllPairs = (iUsefulFeatureCount * iUsefulFeatureCount) * 0.5
    ## (iUsefulFeatureCount * iUsefulFeatureCount) calculates the total number of possible pairs of "useful" features
    ## Multiplying by 0.5 is equivalent to dividing by 2, which accounts for the fact that combinations are used (unordered pairs).
    message("Routing edge calculation for %d possible pairs..." % (iAllPairs))
    lCombinations = itertools.combinations(saUsefulIndices, 2)
    ## itertools.combinations generates all possible combinations of length 2 from the elements in saUsefulIndices.
    ## Each combination represents an unordered pair of indices, which will be used to calculate correlations between pairs of "useful" features.

    # Create queue and threads
    threads = []
    num_worker_threads = THREADS_TO_USE  # DONE: Use available processors
    ## THREADS_TO_USE likely represents the desired number of worker threads to use for parallel processing.
    qCombination = Queue(1000 * num_worker_threads)
    ##This creates a queue (qCombination) with a maximum size of 1000 * num_worker_threads. The queue is used to pass combinations of feature indices to the worker threads for processing.
    
    processes = [Thread(target=addEdgeAboveThreshold, args=(i, qCombination,)) for i in range(num_worker_threads)]
    ## This creates a list of Thread objects (processes), each corresponding to a worker thread.
    ## The target is set to the addEdgeAboveThreshold function, which is the function that will be executed in parallel.
    ## The args parameter is a tuple containing arguments to be passed to the addEdgeAboveThreshold function. In this case, it includes the thread index i and the queue qCombination
    for t in processes:
        t.daemon = True
        #t.setDaemon(True)
        ## This sets each thread in the processes list as a daemon thread. Daemon threads are background threads that are terminated when the main program finishes.
        t.start()
        ## This starts each thread in the processes list, initiating parallel execution of the addEdgeAboveThreshold function.

    # Feed tasks
    iCnt = 1
    dStartTime = perf_counter()
    for iFirstFeatIdx, iSecondFeatIdx in lCombinations:
        qCombination.put((iFirstFeatIdx, iSecondFeatIdx, g, mAllData, saFeatures, iFirstFeatIdx, iSecondFeatIdx,
                          iCnt, iAllPairs, dStartTime, dEdgeThreshold))
        ## This line puts a tuple containing various parameters onto the queue (qCombination)
        ##  this tuple encapsulates all the necessary information for a worker thread to calculate the correlation between two features, determine whether an edge should be added to the graph, and perform the task efficiently. 
        ## The worker threads will dequeue these tuples and execute the corresponding tasks in parallel.
        # Wait a while if we reached full queue
        if qCombination.full():
            message("So far routed %d tasks. Waiting on worker threads to provide more tasks..." % (iCnt))
            time.sleep(0.05)

        iCnt += 1
    message("Routing edge calculation for %d possible pairs... Done." % (iAllPairs))

    message("Waiting for completion...")
    qCombination.join()
   
    ## The qCombination.join() method is used to block the program execution until all tasks in the queue (qCombination) are done. It is typically used in a scenario where multiple threads are performing parallel tasks, 
    ## and the main program needs to wait for all threads to finish their work before proceeding.
    message("Total time (sec): %4.2f" % (perf_counter() - dStartTime))

    message("Creating edges for %d possible pairs... Done." % (iAllPairs))

    message("Extracting graph... Done.")

    message("Removing single nodes... Nodes before removal: %d" % (g.number_of_nodes()))
    toRemove = [curNode for curNode in g.nodes().keys() if len(g[curNode]) == 0]
    ## a list (toRemove) containing the nodes in the graph (g) that have no edges, meaning they are isolated nodes (nodes with degree zero). 
    ## The condition len(g[curNode]) == 0 checks if the node's degree is zero.
    while len(toRemove) > 0:
        g.remove_nodes_from(toRemove) ## This removes the nodes listed in toRemove from the graph g
        toRemove = [curNode for curNode in g.nodes().keys() if len(g[curNode]) == 0] ## After removal, it updates the toRemove list with the names of nodes that are still isolated.
        message("Nodes after removal step: %d" % (g.number_of_nodes()))
    message("Removing single nodes... Done. Nodes after removal: %d" % (g.number_of_nodes()))
    
    message("Main graph edges: " + str(len(g.edges())) +", main graph nodes: " + str(len(g.nodes())))
    message("Saving graph...")
    if stdevFeatSelection:
        write_multiline_adjlist(g, Prefix + "graphSDAdjacencyList.txt") ## save a file using write_multiline_adjlist
        with open(Prefix + "usefulSDFeatureNames.pickle", "wb") as fOut: ## This line opens a file named "usefulFeatureNames.pickle" in binary write mode ("wb"). The with statement is used to ensure that the file is properly closed after writing.
            pickle.dump(saUsefulFeatureNames, fOut)
    else:
        write_multiline_adjlist(g, Prefix + "graphAdjacencyList.txt") ## save a file using write_multiline_adjlist
        with open(Prefix + "usefulFeatureNames.pickle", "wb") as fOut: ## This line opens a file named "usefulFeatureNames.pickle" in binary write mode ("wb"). The with statement is used to ensure that the file is properly closed after writing.
            pickle.dump(saUsefulFeatureNames, fOut) ## serialize the Python object saUsefulFeatureNames and write the serialized data to the file fOut. The object is serialized into a binary format suitable for storage or transmission.

    message("Saving graph... Done.")

    message("Trying to load graph... Done.")

    return g, saUsefulFeatureNames


def getGraphAndData(bResetGraph=False, dEdgeThreshold=0.3, bResetFiles=False, bPostProcessing=True, bstdevFiltering=False, bNormalize=True, bNormalizeLog2Scale=True, bShow = False, 
                    bSave = False, stdevFeatSelection=True, nfeat=50, expSelectedFeats=False, bExportImpMat=False): 
    # TODO: dMinDivergenceToKeep: Add as parameter
    """
    Loads the feature correlation graph and all feature data.
    :param bResetGraph: If True, recalculate graph, else load from disc. Default: False.
    :param dMinDivergenceToKeep: The threshold of data deviation to consider a feature useful. Default: log2(10e6).
    :param dEdgeThreshold: The minimum correlation between features to consider the connection useful. Default: 0.3.
    :param bResetFiles: If True, clear initial feature matrix serialization and re-parse CSV file. Default: False.
    :param bPostProcessing: If True, apply preprocessing to remove NaNs, etc. Default: True.
    :param bNormalize: If True, apply normalization to remove NaNs, etc. Default: True.
    :param bNormalizeLog2Scale: If true, after normalization apply log2 scale to feature values.
    :return: A tuple of the form (feature correlation graph, all feature matrix, instance/case class matrix,
        important feature names list)
    CV update:
    :return: A tuple of the form (feature correlation graph, all feature matrix, instance/case class matrix,
        important feature names list, sample ids)
    """
    # Do mFeatures_noNaNs has all features? Have we applied a threshold to get here?
    mFeatures_noNaNs, vClass, sampleIDs, feat_names, tumor_stage = initializeFeatureMatrices(bResetFiles=bResetFiles, bPostProcessing=bPostProcessing, bstdevFiltering=bstdevFiltering,
                                                         bNormalize=bNormalize, bNormalizeLog2Scale=bNormalizeLog2Scale, nfeat=nfeat, expSelectedFeats=expSelectedFeats, bExportImpMat=bExportImpMat)
    gToDraw, saRemainingFeatureNames = getFeatureGraph(mFeatures_noNaNs, feat_names, dEdgeThreshold=dEdgeThreshold, nfeat=nfeat, bResetGraph=bResetGraph, stdevFeatSelection=stdevFeatSelection)
    
    # if bShow or bSave:
    #     drawAndSaveGraph(gToDraw, sPDFFileName="corrGraph.pdf",bShow = bShow, bSave = bSave)

    return gToDraw, mFeatures_noNaNs, vClass, saRemainingFeatureNames, sampleIDs, feat_names, tumor_stage


def drawAndSaveGraph(gToDraw, sPDFFileName="corrGraph",bShow = True, bSave = True):
    
    """
    Draws and displays a given graph, by using graphviz.
    :param gToDraw: The graph to draw
    """
    if len(gToDraw.edges())<3:
        figure_size = (len(gToDraw.edges()) * 4, len(gToDraw.edges()) * 4)
    else:
        figure_size = (100, 100)
        
    plt.figure(figsize=figure_size)
    # plt.figure(figsize=(len(gToDraw.edges()) , len(gToDraw.edges())))## ru8mizei mege8os figure me bash ton ari8mo ton edges
    plt.clf()

    pos = nx.nx_agraph.graphviz_layout(gToDraw, prog='circo')
    
    try:
        dNodeLabels = {}
        # For each node
        for nCurNode in gToDraw.nodes():
            #!!! Try to add weight
            dNodeLabels[nCurNode] = "%s (%4.2f)" % (str(nCurNode), gToDraw.nodes[nCurNode]['weight'])
            
    except KeyError:
        # Weights could not be added, use nodes as usual
        dNodeLabels = None

    nx.draw_networkx(gToDraw, pos, arrows=False, node_size=1200, node_color="blue", with_labels=True, labels=dNodeLabels)
    ##nx.draw_networkx: Draws the nodes and edges of the graph using the specified positions (pos) and other parameters
    edge_labels = nx.get_edge_attributes(gToDraw, 'weight')
    ##extract the 'weight' attribute from the edges of a NetworkX graph (gToDraw)
    nx.draw_networkx_edge_labels(gToDraw, pos, edge_labels=edge_labels)
    ##nx.draw_networkx_edge_labels: Draws labels for the edges, assuming there are 'weight' attributes associated with the edges

    if bSave:
        message("Saving graph to file...")
        try:
            write_dot(gToDraw, sPDFFileName + '.dot')
            plt.savefig(sPDFFileName + ".pdf", bbox_inches='tight')## bbox_inches='tight': This parameter adjusts the bounding box around the saved figure. The argument 'tight' is used to minimize the whitespace around the actual content of the figure
            # plt.savefig(sPDFFileName)
            message("Saving graph to file... Done.")
        except Exception as e:
            print("Could not save file! Exception:\n%s\n"%(str(e)))
            print("Continuing normally...")
    else:
        message("Ignoring graph saving as requested...")
    if bShow:
        plt.show()

def mGraphDistribution(mFeatures_noNaNs, feat_names, startThreshold = 0.3, endThreshold = 0.8, nfeat=50, bResetGraph=False, stdevFeatSelection=False):
    """
    Plots the distribution of the general graph's edges between start and end thresholds adding by 0.1.
    :param mFeatures_noNaNs: the feature matrix
    :param feat_names: array with the name of the features
    :param startThreshold: the minimum threshold
    :param endThreshold: the maximum threshold
    :param bResetGraph: if True, creates again the graph 
    """
    thresholds = []
    edgesNum = []
    nodesNum = []
    for threshold in np.arange(startThreshold, endThreshold+0.05, 0.1):
        threshold = round(threshold, 1)
        gToDraw, saRemainingFeatureNames = getFeatureGraph(mFeatures_noNaNs, feat_names, dEdgeThreshold=threshold, nfeat=nfeat, bResetGraph=bResetGraph, stdevFeatSelection=stdevFeatSelection)
        thresholds.append(threshold)
        edgesNum.append(gToDraw.number_of_edges())
        nodesNum.append(gToDraw.number_of_nodes())

    #DEBUG LINES
    message(thresholds)
    message(edgesNum)
    #################
    graphData = pd.DataFrame({'thresholds' : thresholds, 'edgesNum': edgesNum, 'nodesNum' : nodesNum})
    plt.clf()
    sns.barplot(graphData, x="thresholds", y="edgesNum")
    
    for i in range(len(thresholds)):
        plt.text(i, edgesNum[i], edgesNum[i], ha = 'center')
    
    plt.xlabel('Pearson correlation thresholds')
    plt.ylabel('Number of edges')
    if stdevFeatSelection:
        plt.title('Number of edges in the main graph from standard deviation \n feature selection')
    else:
        plt.title('Number of edges in the main graph from DEGs')
    plt.show()
    plt.savefig("edgesDistribution.png")

    plt.clf()
    sns.barplot(graphData, x="thresholds", y="nodesNum")
    
    for i in range(len(thresholds)):
        plt.text(i, nodesNum[i], nodesNum[i], ha = 'center')
    
    plt.xlabel('Pearson correlation thresholds')
    plt.ylabel('Number of nodes')
    if stdevFeatSelection:
        plt.title('Number of nodes in the main graph from standard deviation \n feature selection')
    else:
        plt.title('Number of nodes in the main graph from DEGs')
    plt.show()
    plt.savefig("nodesDistribution.png")

def getMeanDegreeCentrality(gGraph):
    """
    Returns the average of the degree centralities of the nodes of a given graph.
    :param gGraph: The given graph.
    :return: The mean of degree centralities.
    """
    mCentralities = list(nx.degree_centrality(gGraph).values())
    return np.mean(mCentralities)


# Does NOT work (for several reasons...)
# def getAvgShortestPath(gGraph):
#     try:
#         fAvgShortestPathLength = nx.algorithms.shortest_paths.average_shortest_path_length(gGraph)
#     except:
#         mShortestPaths = np.asarray(
#             [nx.algorithms.shortest_paths.average_shortest_path_length(g) for g in nx.algorithms.components.connected.connected_components(gGraph)])
#         fAvgShortestPathLength = np.mean(mShortestPaths)
#
#     return fAvgShortestPathLength

def avg_shortest_path(gGraph):
    res=[]
    connected_components = nx.connected_components(gGraph)
    for component  in connected_components:
    
        for node in component:
            new_set = component.copy()
            new_set.remove(node)
            
            for targetNode in new_set:
                res.append(nx.shortest_path_length(gGraph, source=node, target=targetNode))
    return np.average(res)

def getGraphVector(gGraph):
    """
    Represents a given graph as a vector/matrix, where each feature represents a graph description metric.
    :param gGraph: The graph to represent.
    :return: The feature vector, consisting of: #edges,#nodes, mean node degree centrality, number of cliques,
    average node connectivity, mean pair-wise shortest paths of connected nodes.
    """
    # DEBUG LINES
    message("Extracting graph feature vector...")
   
   
    mRes = np.asarray(
        [len(gGraph.edges()), len(gGraph.nodes()),
        np.mean(np.array(list(nx.algorithms.centrality.degree_alg.degree_centrality(gGraph).values()))),
        len(list(nx.find_cliques(gGraph))),
        nx.algorithms.connectivity.connectivity.average_node_connectivity(gGraph),
        avg_shortest_path(gGraph)
        ])
        
    # DEBUG LINES
    message("Extracting graph feature vector... Done.")

    return mRes

def spreadingActivation(gGraph, iIterations=100, dPreservationPercent=0.5, bAbsoluteMass=False):
    """
    Applies spreading activation to a given graph.
    :param gGraph: The graph used to apply spreading activation.
    :param iIterations: The number of iterations for the spreading.
    :param dPreservationPercent: The preservation of mass from each node, during the spreading.
    :param bAbsoluteMass: If True, use absolute values of mass. Otherwise, also allow negative spreading.
    :return: The (inplace) updated graph.
    """
    message("Applying spreading activation...")
    #!!! In each iteration
    for iIterCnt in range(iIterations):
        #!!! For every node
        for nCurNode in gGraph.nodes():
            # Get max edge weight
            dWeights = np.asarray([gGraph[nCurNode][nNeighborNode]['weight'] for nNeighborNode in gGraph[nCurNode]])
        
            dWeightSum = np.sum(dWeights)
            # For every neighbor
            for nNeighborNode in gGraph[nCurNode]:
                # Get edge percantile weight
                dMassPercentageToMove = gGraph[nCurNode][nNeighborNode]['weight'] / dWeightSum
                
                try:
                    # Assign part of the weight to the neighbor
                    dMassToMove = (1.0 - dPreservationPercent) * gGraph.nodes[nCurNode][
                        'weight'] * dMassPercentageToMove
                    
                    # Work with absolute numbers, if requested
                    if bAbsoluteMass:
                        gGraph.nodes[nNeighborNode]['weight'] = abs(gGraph.nodes[nNeighborNode]['weight']) + abs(
                            dMassToMove)
                    else:
                        gGraph.nodes[nNeighborNode]['weight'] += dMassToMove
                except KeyError:
                    message("Warning: node %s has no weight assigned. Assigning 0." % (str(nCurNode)))
                    gGraph.nodes[nNeighborNode]['weight'] = 0

            # Reduce my weight equivalently
            gGraph.nodes[nCurNode]['weight'] *= dPreservationPercent
            
    message("Applying spreading activation... Done.")
    return gGraph


def assignSampleValuesToGraphNodes(gGraph, mSample, saSampleFeatureNames, feat_names):
    """
    Assigns values/weights to nodes of a given graph (inplace), for a given sample.
    :param gGraph: The generic graph.
    :param mSample: The sample which will define the feature node values/weights.
    :param saSampleFeatureNames: The mapping between feature names and indices.
    """
    # For each node
    for nNode in gGraph.nodes():
        # Get corresponding feature idx in sample 
        iFeatIdx = feat_names.index(nNode)
        # Assign value of feature as node weight
        dVal = mSample[iFeatIdx]
        # Handle missing values as zero (i.e. non-important)
        if dVal == np.NAN:
            dVal = 0

        gGraph.nodes[nNode]['weight'] = dVal


def filterGraphNodes(gMainGraph, dKeepRatio):
    """
    Filters elements of a given graph (inplace), keeping a ratio of the top nodes, when ordered
    descending based on weight/value.

    :param gMainGraph: The graph from which nodes will be removed/filtered.
    :param dKeepRatio: The ratio of nodes we want to keep (between 0.0 and 1.0).
    :return: The filtered graph.
    """
    # Get all weights
    mWeights = np.asarray([gMainGraph.nodes[curNode]['weight'] for curNode in gMainGraph.nodes().keys()])
    
    # DEBUG LINES
    #message("mWeights: "+str(mWeights))
    message("Filtering nodes... Weights: %s"%(str(mWeights.shape)))
    # If empty weights (possibly because the threshold is too high
    if (mWeights.shape[0] == 0):
        # Update the user and continue
        message("WARNING: The graph is empty...")
        
    ##########
    # Find appropriate percentile
    dMinWeight = np.percentile(mWeights, (1.0 - dKeepRatio) * 100)
    # Select and remove nodes with lower value
    toRemove = [curNode for curNode in gMainGraph.nodes().keys() if gMainGraph.nodes[curNode]['weight'] < dMinWeight]
    gMainGraph.remove_nodes_from(toRemove)

    return gMainGraph
        

def generateAllSampleGraphFeatureVectors(gMainGraph, mAllSamples, saRemainingFeatureNames, sampleIDs, feat_names, bShowGraphs, bSaveGraphs):
    """
    Generates graph feature vectors for all samples and returns them as a matrix.
    :param gMainGraph: The generic graph of feature correlations.
    :param mAllSamples: The samples to uniquely represent as graph feature vectors.
    :param saRemainingFeatureNames: The useful features subset.
    :return: A matrix representing the samples (rows), based on their graph representation.
    """
    ########################
    # Create queue and threads
    threads = []
    num_worker_threads = THREADS_TO_USE 
    qTasks = Queue(10 * num_worker_threads) 
    

    # Count instances
    iAllCount = np.shape(mAllSamples)[0] 

    # Item iterator
    iCnt = iter(range(1, iAllCount + 1)) 
    dStartTime = perf_counter()

    # Init result list
    dResDict = {}
    graphList = []
    # Counter for the specific sampleID suffix
    saveCounter = {"11A": 0, "01A": 0} 
    
    
    threads = [Thread(target=getSampleGraphFeatureVector, args=(i, qTasks,bShowGraphs, bSaveGraphs,)) for i in range(num_worker_threads)]
    for t in threads:
        t.daemon = True 
        t.start() 
    
    # Add all items to queue
    for idx in range (np.shape(mAllSamples)[0]):
        qTasks.put((sampleIDs[idx], dResDict, gMainGraph, mAllSamples[idx, :], saRemainingFeatureNames, feat_names, next(iCnt), iAllCount, dStartTime, saveCounter, graphList))
    
    message("Waiting for completion...")
    
    qTasks.join() 

    message("Total time (sec): %4.2f" % (perf_counter() - dStartTime))

    # Plot and save the collected graphs
    # for gMainGraph, sPDFFileName in graphList:
    #     drawAndSaveGraph(gMainGraph, sPDFFileName, bShowGraphs, bSaveGraphs)

    return dResDict


def getSampleGraphFeatureVector(i, qQueue, bShowGraphs=True, bSaveGraphs=True):
    """
    Helper parallelization function, which calculates the graph representation of a given sample.
    :param i: The thread number calling the helper.
    :param qQueue: A Queue, from which the execution data will be drawn. Should contain:
    dResDict -- reference to the dictionary containing the result
    gMainGraph -- the generic graph of feature correlations
    mSample -- the sample to represent
    saRemainingFeatureNames -- the list of useful feature names
    iCnt -- the current sample count
    iAllCount -- the number of all samples to be represented
    dStartTime -- the time when parallelization started
    """
    # dSample = {}

    iWaitingCnt = 0 # Number of tries, finding empty queue
    while True:
        try:
            params = qQueue.get_nowait()
        
        except Empty:
            if iWaitingCnt < 3:
                message("Found no items... Waiting... (already waited %d times)"%(iWaitingCnt))
                time.sleep(1)
                iWaitingCnt += 1 # Waited one more time
                continue
                 
            message("Waited long enough. Reached and of queue... Stopping.")
            break
        
        sampleID, dResDict, gMainGraph, mSample, saRemainingFeatureNames, feat_names, iCnt, iAllCount, dStartTime, saveCounter, graphList = params
           
        # DEBUG LINES  
        message("Working on instance %d of %d..." % (iCnt, iAllCount))
        #############

        # Create a copy of the graph
        gMainGraph = copy.deepcopy(gMainGraph)

        # Assign values    
        assignSampleValuesToGraphNodes(gMainGraph, mSample, saRemainingFeatureNames, feat_names)
        # Apply spreading activation
        gMainGraph = spreadingActivation(gMainGraph, bAbsoluteMass=True)  # TODO: Add parameter, if needed
        # Keep top performer nodes
        gMainGraph = filterGraphNodes(gMainGraph, dKeepRatio=0.25)  # TODO: Add parameter, if needed
        # Extract and return features
        vGraphFeatures = getGraphVector(gMainGraph)
        
        # Save or show the graph if required
        if sampleID.endswith("01A") or sampleID.endswith("11A"):
            suffix = sampleID[-3:]  # Extract the suffix (last 3 characters)
            if saveCounter[suffix] < 2:
                saveCounter[suffix] += 1
                with lock:
                    graphList.append((gMainGraph, "graph_" + sampleID))


        #DEBUGLINES
        #message("Calling drawAndSaveGraph for graph %s..."%(str(sampleID)))
        #if not exists("/home/thlamp/scripts/testcorrSample.pdf"):
        #    drawAndSaveGraph(gMainGraph, sPDFFileName = "testcorrSample.pdf", bShow = bShowGraphs, bSave = bSaveGraphs)
        #message("Calling drawAndSaveGraph...Done")
        ######################

        #  Add to common result queue
        
        #with lock:  # Acquire the lock before modifying the shared resource
        dResDict[sampleID] = vGraphFeatures
        
        # Signal done
        qQueue.task_done()

        # DEBUG LINES
        if iCnt % 5 == 0 and (iCnt != 0):
            dNow = perf_counter()
            dRate = ((dNow - dStartTime) / iCnt)
            dRemaining = (iAllCount - iCnt) * dRate
            message("%d (Estimated remaining (sec): %4.2f - Working at a rate of %4.2f samples/sec)\n" % (
                iCnt, dRemaining, 1.0 / dRate))

# def classify(X, y, lmetricResults, sfeatClass):
#     """
#     Calculates and outputs the performance of classification, through Leave-One-Out cross-valuation, given a set of feature vectors and a set of labels.
#     :param X: The feature vector matrix.
#     :param y: The labels.
#     :param lmetricResults: list for the results of performance metrics.
#     :param sfeatClass: string/information about the ML model, the features and data labels 
#     """
#     scoring = {
#     'accuracy': make_scorer(accuracy_score),
#     'f1_micro': make_scorer(f1_score, average="micro"),
#     'f1_macro': make_scorer(f1_score, average="macro")}

#     classifier = DecisionTreeClassifier()
    
#     cv = LeaveOneOut() 

#     # Calculate cross-validation scores for both accuracy and F1
#     scores = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
    
#     # Calculate SEM 
#     sem_accuracy = np.std(scores['test_accuracy']) / np.sqrt(len(scores['test_accuracy']))
#     sem_f1_micro = np.std(scores['test_f1_micro']) / np.sqrt(len(scores['test_f1_micro']))
#     sem_f1_macro = np.std(scores['test_f1_macro']) / np.sqrt(len(scores['test_f1_macro']))

#     message("Avg. Performanace: %4.2f (st. dev. %4.2f, sem %4.2f) \n %s" % (np.mean(scores['test_accuracy']), np.std(scores['test_accuracy']), sem_accuracy, str(scores['test_accuracy'])))
#     message("Avg. F1-micro: %4.2f (st. dev. %4.2f, sem %4.2f) \n %s" % (np.mean(scores['test_f1_micro']), np.std(scores['test_f1_micro']), sem_f1_micro, str(scores['test_f1_micro'])))
#     message("Avg. F1-macro: %4.2f (st. dev. %4.2f, sem %4.2f) \n %s" % (np.mean(scores['test_f1_macro']), np.std(scores['test_f1_macro']), sem_f1_macro, str(scores['test_f1_macro'])))
    
#     lmetricResults.append([sfeatClass, np.mean(scores['test_accuracy']), sem_accuracy, np.mean(scores['test_f1_micro']), sem_f1_micro, 
#                       np.mean(scores['test_f1_macro']), sem_f1_macro])
#     # Output model
#     classifier.fit(X, y)
#     dot_data = tree.export_graphviz(classifier, out_file=None)
#     graph = graphviz.Source(dot_data)
#     graph.render("Rules")

def classify(X, y, lmetricResults, sfeatClass, savedResults):
    """
    Calculates and outputs the performance of classification, through Leave-One-Out cross-valuation, given a set of feature vectors and a set of labels.
    :param X: The feature vector matrix.
    :param y: The labels.
    :param lmetricResults: list for the results of performance metrics.
    :param sfeatClass: string/information about the ML model, the features and data labels 
    :param savedResults: dictionary for the F1-macro results for wilcoxon test
    """

    classifier = DecisionTreeClassifier(class_weight="balanced")
    
    cv = StratifiedKFold(n_splits=10)
    crossValidation(X, y, cv, classifier, lmetricResults, sfeatClass, savedResults)


def stratifiedDummyClf(X, y, lmetricResults, sfeatClass, savedResults):
    """
    Calculates and outputs the performance of classification, through Leave-One-Out cross-valuation, given a set of feature vectors and a set of labels.
    :param X: The feature vector matrix.
    :param y: The labels.
    :param lmetricResults: list for the results of performance metrics.
    :param sfeatClass: string/information about the ML model, the features and data labels
    :param savedResults: dictionary for the F1-macro results for wilcoxon test 
    """
    dummy_clf = DummyClassifier(strategy="stratified")
    
    cv = StratifiedKFold(n_splits=10)
    crossValidation(X, y, cv, dummy_clf, lmetricResults, sfeatClass, savedResults)

def mostFrequentDummyClf(X, y, lmetricResults, sfeatClass, savedResults):
    """
    Calculates and outputs the performance of classification, through Leave-One-Out cross-valuation, given a set of feature vectors and a set of labels.
    :param X: The feature vector matrix.
    :param y: The labels.
    :param lmetricResults: list for the results of performance metrics.
    :param sfeatClass: string/information about the ML model, the features and data labels
    :param savedResults: dictionary for the F1-macro results for wilcoxon test 
    """
    dummy_clf = DummyClassifier(strategy="most_frequent")
    
    cv = StratifiedKFold(n_splits=10)
    crossValidation(X, y, cv, dummy_clf, lmetricResults, sfeatClass, savedResults)

def mlpClassifier(X, y, lmetricResults, sfeatClass, savedResults):
    """
    Calculates and outputs the performance of classification, through Leave-One-Out cross-valuation, given a set of feature vectors and a set of labels.
    :param X: The feature vector matrix.
    :param y: The labels.
    :param lmetricResults: list for the results of performance metrics.
    :param sfeatClass: string/information about the ML model, the features and data labels 
    :param savedResults: dictionary for the F1-macro results for wilcoxon test
    """
    clf = MLPClassifier(max_iter=2000)

    cv = StratifiedKFold(n_splits=10)
    crossValidation(X, y, cv, clf, lmetricResults, sfeatClass, savedResults)
    

# def crossValidation(X, y, cv, model, lmetricResults, sfeatClass):
#     """
#     Performs the cross validation and save the metrics per iteration, computes the overall matrics and plot the confusion matrix
#     :param X: The feature vector matrix.
#     :param y: The labels.
#     :param cv: the fold that were created from cross validation
#     :param lmetricResults: list for the results of performance metrics.
#     :param sfeatClass: string/information about the ML model, the features and data labels 
#     """
#     # Initialize lists to store metrics per fold
#     accuracy_per_fold = []
#     f1_macro_per_fold = []
#     f1_micro_per_fold = []
#     final_y_pred = []

#     #DEBUG LINES
#     #test = 0
#     ##########

#     # Perform cross-validation
#     for train_index, test_index in cv.split(X):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
    
#         # Fit the classifier on the training data
#         model.fit(X_train, y_train)

#         # Predict label for the test data
#         y_pred = model.predict(X_test)

#         # Calculate metrics for this fold
#         accuracy = accuracy_score(y_test, y_pred)
#         f1_macro = f1_score(y_test, y_pred, average='macro')
#         f1_micro = f1_score(y_test, y_pred, average='micro')
#         print(f1_macro)
#         final_y_pred.append(y_pred[0])

#         # Append metrics to lists
#         accuracy_per_fold.append(accuracy)
#         f1_macro_per_fold.append(f1_macro)
#         f1_micro_per_fold.append(f1_micro)
    
    # accuracy = accuracy_score(y, final_y_pred)
    # f1_micro = f1_score(y, final_y_pred, average='micro')
    # f1_macro = f1_score(y, final_y_pred, average='macro')
    
    #DEBUG LINES
    # message(accuracy)
    # message(f1_micro)
    # message(f1_macro)
    # message(f1_macro_per_fold)
    ###############
    
    # Calculate SEM 
    # sem_accuracy = np.std(accuracy_per_fold) / np.sqrt(len(accuracy_per_fold))
    # sem_f1_micro = np.std(f1_micro_per_fold) / np.sqrt(len(f1_micro_per_fold))
    # sem_f1_macro = np.std(f1_macro_per_fold) / np.sqrt(len(f1_macro_per_fold))

    # message("Avg. Performanace: %4.2f (st. dev. %4.2f, sem %4.2f) \n %s" % (np.mean(accuracy_per_fold), np.std(accuracy_per_fold), sem_accuracy, str(accuracy_per_fold)))
    # message("Avg. F1-micro: %4.2f (st. dev. %4.2f, sem %4.2f) \n %s" % (f1_micro, np.std(f1_micro_per_fold), sem_f1_micro, str(f1_micro_per_fold)))
    # message("Avg. F1-macro: %4.2f (st. dev. %4.2f, sem %4.2f) \n %s" % (f1_macro, np.std(f1_macro_per_fold), sem_f1_macro, str(f1_macro_per_fold)))

def crossValidation(X, y, cv, model, lmetricResults, sfeatClass, savedResults): 
    """
    Performs the cross validation and save the metrics per iteration, computes the overall matrics and plot the confusion matrix
    :param X: The feature vector matrix.
    :param y: The labels.
    :param cv: the fold that were created from cross validation
    :param lmetricResults: list for the results of performance metrics.
    :param sfeatClass: string/information about the ML model, the features and data labels 
    :param savedResults: dictionary for the F1-macro results for wilcoxon test
    """
    # Initialize lists to store metrics per fold
    accuracy_per_fold = []
    f1_macro_per_fold = []
    f1_micro_per_fold = []
    final_y_pred = []
    final_y = []    
    # Perform cross-validation
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
       
        final_y.extend(y_test)

        # Fit the classifier on the training data
        model.fit(X_train, y_train)

        # Predict label for the test data
        y_pred = model.predict(X_test)
        
        # Calculate metrics for this fold
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')

        final_y_pred.extend(y_pred)

        # Append metrics to lists
        accuracy_per_fold.append(accuracy)
        f1_macro_per_fold.append(f1_macro)
        f1_micro_per_fold.append(f1_micro)

    # Calculate SEM 
    sem_accuracy = np.std(accuracy_per_fold) / np.sqrt(len(accuracy_per_fold))
    sem_f1_micro = np.std(f1_micro_per_fold) / np.sqrt(len(f1_micro_per_fold))
    sem_f1_macro = np.std(f1_macro_per_fold) / np.sqrt(len(f1_macro_per_fold))  

    message("Avg. Accuracy: %4.2f (st. dev. %4.2f, sem %4.2f)" % (np.mean(accuracy_per_fold), np.std(accuracy_per_fold), sem_accuracy))#\n %s      , str(accuracy_per_fold)
    message("Avg. F1-micro: %4.2f (st. dev. %4.2f, sem %4.2f)" % (np.mean(f1_micro_per_fold), np.std(f1_micro_per_fold), sem_f1_micro))# \n %s     , str(f1_micro_per_fold)
    message("Avg. F1-macro: %4.2f (st. dev. %4.2f, sem %4.2f)\n %s" % (np.mean(f1_macro_per_fold), np.std(f1_macro_per_fold), sem_f1_macro, str(f1_macro_per_fold)))# \n %s     , str(f1_macro_per_fold)
    
    savedResults[sfeatClass]={}
    savedResults[sfeatClass]["mean_accuracy"]=np.mean(accuracy_per_fold)
    savedResults[sfeatClass]["mean_F1_micro"]=np.mean(f1_micro_per_fold)
    savedResults[sfeatClass]["mean_F1_macro"]=np.mean(f1_macro_per_fold)

    savedResults[sfeatClass]["std_accuracy"]=np.std(accuracy_per_fold)
    savedResults[sfeatClass]["std_F1_micro"]=np.std(f1_micro_per_fold)
    savedResults[sfeatClass]["std_F1_macro"]=np.std(f1_macro_per_fold)

    savedResults[sfeatClass]["sem_accuracy"]=sem_accuracy
    savedResults[sfeatClass]["sem_F1_micro"]=sem_f1_micro
    savedResults[sfeatClass]["sem_F1_macro"]=sem_f1_macro
    
    savedResults[sfeatClass]["accuracy_per_fold"]=accuracy_per_fold
    savedResults[sfeatClass]["f1_micro_per_fold"]=f1_micro_per_fold
    savedResults[sfeatClass]["f1_macro_per_fold"]=f1_macro_per_fold


    cm = confusion_matrix(final_y, final_y_pred)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("confMat"+ sfeatClass +".png")
    plt.show()

    lmetricResults.append([sfeatClass, np.mean(accuracy_per_fold), sem_accuracy, np.mean(f1_micro_per_fold), sem_f1_micro, np.mean(f1_macro_per_fold), sem_f1_macro])

def xgboost(X, y, lmetricResults, sfeatClass, savedResults):
    """
    Calculates and outputs the performance of classification, through Leave-One-Out cross-valuation, given a set of feature vectors and a set of labels.
    :param X: The feature vector matrix.
    :param y: The labels.
    :param lmetricResults: list for the results of performance metrics.
    :param sfeatClass: string/information about the ML model, the features and data labels 
    :param savedResults: dictionary for the F1-macro results for wilcoxon test
    """
    model = xgb.XGBClassifier()
    
    # scoring = {
    # 'accuracy': make_scorer(accuracy_score),
    # 'f1_micro': make_scorer(f1_score, average="micro"),
    # 'f1_macro': make_scorer(f1_score, average="macro")}
    
    cv = StratifiedKFold(n_splits=10)
    crossValidation(X, y, cv, model, lmetricResults, sfeatClass, savedResults)

    # # Calculate cross-validation scores for both accuracy and F1
    # scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    
    # # Calculate SEM 
    # sem_accuracy = np.std(scores['test_accuracy']) / np.sqrt(len(scores['test_accuracy']))
    # sem_f1_micro = np.std(scores['test_f1_micro']) / np.sqrt(len(scores['test_f1_micro']))
    # sem_f1_macro = np.std(scores['test_f1_macro']) / np.sqrt(len(scores['test_f1_macro']))

    # print("Avg. Performanace: %4.2f (st. dev. %4.2f, sem %4.2f) \n %s" % (np.mean(scores['test_accuracy']), np.std(scores['test_accuracy']), sem_accuracy, str(scores['test_accuracy'])))
    # print("Avg. F1-micro: %4.2f (st. dev. %4.2f, sem %4.2f) \n %s" % (np.mean(scores['test_f1_micro']), np.std(scores['test_f1_micro']), sem_f1_micro, str(scores['test_f1_micro'])))
    # print("Avg. F1-macro: %4.2f (st. dev. %4.2f, sem %4.2f) \n %s" % (np.mean(scores['test_f1_macro']), np.std(scores['test_f1_macro']), sem_f1_macro, str(scores['test_f1_macro'])))
    
    # lmetricResults.append([sfeatClass, np.mean(scores['test_accuracy']), sem_accuracy, np.mean(scores['test_f1_micro']), sem_f1_micro, 
    #                   np.mean(scores['test_f1_macro']), sem_f1_macro])


def RandomForest(X, y, lmetricResults, sfeatClass, savedResults):
    """
    Calculates and outputs the performance of classification, through Leave-One-Out cross-valuation, given a set of feature vectors and a set of labels.
    :param X: The feature vector matrix.
    :param y: The labels.
    :param lmetricResults: list for the results of performance metrics.
    :param sfeatClass: string/information about the ML model, the features and data labels 
    :param savedResults: dictionary for the F1-macro results for wilcoxon test
    """
    clf = RandomForestClassifier(class_weight = "balanced")
    
    # scoring = {
    # 'accuracy': make_scorer(accuracy_score),
    # 'f1_micro': make_scorer(f1_score, average="micro"),
    # 'f1_macro': make_scorer(f1_score, average="macro")}
    
    cv = StratifiedKFold(n_splits=10) 
    crossValidation(X, y, cv, clf, lmetricResults, sfeatClass, savedResults)
    # Calculate cross-validation scores for both accuracy and F1
    # scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)
    
    # Calculate SEM 
    # sem_accuracy = np.std(scores['test_accuracy']) / np.sqrt(len(scores['test_accuracy']))
    # sem_f1_micro = np.std(scores['test_f1_micro']) / np.sqrt(len(scores['test_f1_micro']))
    # sem_f1_macro = np.std(scores['test_f1_macro']) / np.sqrt(len(scores['test_f1_macro']))

    # print("Avg. Performanace: %4.2f (st. dev. %4.2f, sem %4.2f) \n %s" % (np.mean(scores['test_accuracy']), np.std(scores['test_accuracy']), sem_accuracy, str(scores['test_accuracy'])))
    # print("Avg. F1-micro: %4.2f (st. dev. %4.2f, sem %4.2f) \n %s" % (np.mean(scores['test_f1_micro']), np.std(scores['test_f1_micro']), sem_f1_micro, str(scores['test_f1_micro'])))
    # print("Avg. F1-macro: %4.2f (st. dev. %4.2f, sem %4.2f) \n %s" % (np.mean(scores['test_f1_macro']), np.std(scores['test_f1_macro']), sem_f1_macro, str(scores['test_f1_macro'])))
    
    # lmetricResults.append([sfeatClass, np.mean(scores['test_accuracy']), sem_accuracy, np.mean(scores['test_f1_micro']), sem_f1_micro, 
    #                   np.mean(scores['test_f1_macro']), sem_f1_macro])

def NBayes(X, y, lmetricResults, sfeatClass, savedResults):
    """
    Calculates and outputs the performance of classification, through Leave-One-Out cross-valuation, given a set of feature vectors and a set of labels.
    :param X: The feature vector matrix.
    :param y: The labels.
    :param lmetricResults: list for the results of performance metrics.
    :param sfeatClass: string/information about the ML model, the features and data labels 
    :param savedResults: dictionary for the F1-macro results for wilcoxon test
    """
    gnb = GaussianNB()
    
    # scoring = {
    # 'accuracy': make_scorer(accuracy_score),
    # 'f1_micro': make_scorer(f1_score, average="micro"),
    # 'f1_macro': make_scorer(f1_score, average="macro")}
    
    cv = StratifiedKFold(n_splits=10)
    crossValidation(X, y, cv, gnb, lmetricResults, sfeatClass, savedResults)
    # Calculate cross-validation scores for both accuracy and F1
    # scores = cross_validate(gnb, X, y, cv=cv, scoring=scoring)
    
    # Calculate SEM 
    # sem_accuracy = np.std(scores['test_accuracy']) / np.sqrt(len(scores['test_accuracy']))
    # sem_f1_micro = np.std(scores['test_f1_micro']) / np.sqrt(len(scores['test_f1_micro']))
    # sem_f1_macro = np.std(scores['test_f1_macro']) / np.sqrt(len(scores['test_f1_macro']))

    # print("Avg. Performanace: %4.2f (st. dev. %4.2f, sem %4.2f) \n %s" % (np.mean(scores['test_accuracy']), np.std(scores['test_accuracy']), sem_accuracy, str(scores['test_accuracy'])))
    # print("Avg. F1-micro: %4.2f (st. dev. %4.2f, sem %4.2f) \n %s" % (np.mean(scores['test_f1_micro']), np.std(scores['test_f1_micro']), sem_f1_micro, str(scores['test_f1_micro'])))
    # print("Avg. F1-macro: %4.2f (st. dev. %4.2f, sem %4.2f) \n %s" % (np.mean(scores['test_f1_macro']), np.std(scores['test_f1_macro']), sem_f1_macro, str(scores['test_f1_macro'])))
    
    # lmetricResults.append([sfeatClass, np.mean(scores['test_accuracy']), sem_accuracy, np.mean(scores['test_f1_micro']), sem_f1_micro, 
    #                   np.mean(scores['test_f1_macro']), sem_f1_macro])


def getSampleGraphVectors(gMainGraph, mFeatures_noNaNs, saRemainingFeatureNames, sampleIDs, feat_names, bResetFeatures=True, dEdgeThreshold=0.3, nfeat=50,
                          numOfSelectedSamples=-1, bShowGraphs=True, bSaveGraphs=True, stdevFeatSelection=True):
    """
    Extracts the graph feature vectors of a given set of instances/cases.
    :param gMainGraph: The overall feature correlation graph.
    :param mFeatures_noNaNs: The (clean from NaNs) feature matrix of instances/cases.
    :param saRemainingFeatureNames: The list of useful feature names.
    :param bResetFeatures: If True, features will be re-calculated. Otherwise, they will be loaded from an intermediate
    file. Default: True.
    :param numOfSelectedSamples: Allows working on a subset of the data. If -1, then use all data. Else use the given
    number of instances (half of which are taken from the first instances in mFeatures_noNaNs, while half from the
    last ones). Default: -1 (i.e. all samples).
    :return: A matrix containing the graph feature vectors of the selected samples.
    """
    # Get all sample graph vectors
    try:
        message("Trying to load graph feature matrix...")
        if bResetFeatures:
            raise Exception("User requested rebuild of features.")
        if stdevFeatSelection:
            with open(Prefix + "SDgraphFeatures_" + str(dEdgeThreshold) + "_" + str(nfeat) + ".pickle", "rb") as fIn:
                mGraphFeatures = pickle.load(fIn)
        else:
            with open(Prefix + "graphFeatures_" + str(dEdgeThreshold) + "_" + str(nfeat) + ".pickle", "rb") as fIn:
                mGraphFeatures = pickle.load(fIn)
        message("Trying to load graph feature matrix... Done.")
    except Exception as e:
        message("Trying to load graph feature matrix... Failed:\n%s" % (str(e)))
        message("Computing graph feature matrix...")

        if (numOfSelectedSamples < 0): 
            mSamplesSelected = mFeatures_noNaNs
            sampleIDsSelected = sampleIDs
        else:
            mSamplesSelected = np.concatenate((mFeatures_noNaNs[0:int(numOfSelectedSamples / 2)][:], 
                                               mFeatures_noNaNs[-int(numOfSelectedSamples / 2):][:]), axis=0) 
            sampleIDsSelected = np.concatenate((sampleIDs[0:int(numOfSelectedSamples / 2)],sampleIDs[-int(numOfSelectedSamples / 2):]))
            
        message("Extracted selected samples:\n" + str(mSamplesSelected[:][0:10]))
        # Extract vectors
        # TODO pass SampleID to generateAllSampleGraphFeatureVectors
        dResDict = generateAllSampleGraphFeatureVectors(gMainGraph, mSamplesSelected, saRemainingFeatureNames, sampleIDsSelected, feat_names, bShowGraphs, bSaveGraphs)
        
        mGraphFeatures = np.array(list(dResDict.values())) 
        reorderedSampleIds = np.array(list(dResDict.keys()))

        # Create a mapping from reorderedSampleIds to their positions
        index_map = {id_: idx for idx, id_ in enumerate(reorderedSampleIds)}
        
        # Find the indices that would reorder reorderedSampleIds to match sampleIDsSelected
        order_indices = [index_map[id_] for id_ in sampleIDsSelected]
        

        # Reorder mGraphFeatures using the calculated indices
        mGraphFeatures = mGraphFeatures[order_indices]

        #DEBUG LINES
        message("dResDict: " + str(dResDict))
        message("mGraphFeatures: " + str(mGraphFeatures))
        ############

        message("Computing graph feature matrix... Done.")

        message("Saving graph feature matrix...")
        if stdevFeatSelection:
            with open(Prefix + "SDgraphFeatures_" + str(dEdgeThreshold) + "_" + str(nfeat) + ".pickle", "wb") as fOut:
                pickle.dump(mGraphFeatures, fOut)  
        else:
            with open(Prefix + "graphFeatures_" + str(dEdgeThreshold) + "_" + str(nfeat) + ".pickle", "wb") as fOut:
                pickle.dump(mGraphFeatures, fOut) 
        message("Saving graph feature matrix... Done.")
    return mGraphFeatures

def getDegs():
    """
    Reads the csv file with the DEGs from R script.
    :return: an array with the name of the DEGs 
    """
    fUsefulFeatureNames = open("/home/thlamp/tcga/bladder_results/DEGs.csv", "r")

    # labelfile, should have stored tumor_stage or labels?       

    saUsefulFeatureNames = np.genfromtxt(fUsefulFeatureNames, skip_header=1, usecols=(0),
                                    missing_values=['NA', "na", '-', '--', 'n/a'],
                                    dtype=np.dtype("object"), delimiter=',').astype(str)
    ##numpy.genfromtxt function to read data from a file. This function is commonly used to load data from text files into a NumPy array.
    ##dtype=np.dtype("object"): This sets the data type for the resulting NumPy array to "object," which is a generic data type that can hold any type of data

    #+ removes " from first column 
    saUsefulFeatureNames[:] = np.char.replace(saUsefulFeatureNames[:], '"', '')

    fUsefulFeatureNames.close()
    return saUsefulFeatureNames

def wilcoxonTests(metricsResults):
    """
    Function that performs wilcoxon test for each pair of ML models F1-macro results
    :param metricsResults: dictionary with the F1-macro results
    """
    keys = list(metricsResults.keys())
    n = len(keys)
    
    for i in range(n):
        for j in range(i + 1, n):
            key1 = keys[i]
            key2 = keys[j]
            data1 = metricsResults[key1]
            data2 = metricsResults[key2]
            
            stat, p = wilcoxon(data1, data2)
            
            print(f"Wilcoxon test between {key1} and {key2}:")
            print(f"Statistic: {stat}, p-value: {p}\n")

def scalingPerClass(matrix, classes):
    # Separate the matrix into two sub-matrices based on class labels
    matrix_class_0 = matrix[classes == "1"]
    matrix_class_1 = matrix[classes == "2"]

    # Initialize the StandardScaler
    scaler_0 = StandardScaler()
    scaler_1 = StandardScaler()

    # Scale each sub-matrix
    scaled_matrix_class_0 = scaler_0.fit_transform(matrix_class_0)
    # DEBUG LINES
    message("Shape of non control matrix in scaling: " + str(np.shape(scaled_matrix_class_0)))
    ###########
    scaled_matrix_class_1 = scaler_1.fit_transform(matrix_class_1)
    # DEBUG LINES
    message("Shape of control matrix in scaling: " + str(np.shape(scaled_matrix_class_1)))
    ###########

    # Combine the scaled sub-matrices back into the original positions
    scaled_matrix = np.zeros_like(matrix, dtype=float)
    
    scaled_matrix[classes == "1"] = scaled_matrix_class_0
    scaled_matrix[classes == "2"] = scaled_matrix_class_1

    # search for columns that have only 0
    res = np.all(scaled_matrix == 0, axis = 0)
    # keep the indices from the columns except from these with only 0
    resIndex = np.where(~res)[0]
    #DEBUG LINES
    print(resIndex)
    ############
    # remove the columns that have only 0 from the graph matrix
    scaled_matrix = scaled_matrix[:, resIndex]
    return scaled_matrix

def main(argv):
    # Init arguments
    parser = argparse.ArgumentParser(description='Perform bladder tumor analysis experiments.')

    # File caching / intermediate files
    parser.add_argument("-rc", "--resetCSVCacheFiles", action="store_true", default=False)
    parser.add_argument("-rg", "--resetGraph", action="store_true", default=False)
    parser.add_argument("-rf", "--resetFeatures", action="store_true", default=False)
    parser.add_argument("-pre", "--prefixForIntermediateFiles", default="")
    # Graph saving and display
    parser.add_argument("-savg", "--saveGraphs", action="store_true", default=False)
    parser.add_argument("-shg", "--showGraphs", action="store_true", default=False)

    # Post-processing control
    parser.add_argument("-p", "--postProcessing", action="store_true", default=False)  # If False NO postprocessing occurs
    parser.add_argument("-norm", "--normalization", action="store_true", default=False)
    parser.add_argument("-ls", "--logScale", action="store_true", default=False)
    parser.add_argument("-stdf", "--stdevFiltering", action="store_true", default=False)
    parser.add_argument("-nfeat", "--numberOfFeaturesPerLevel", type=int, default=50)

    # Post-processing graph features
    parser.add_argument("-scalDeact", "--scalingDeactivation", action="store_false", default=True)
    # parser.add_argument("-scalCls", "--scalingClass", action="store_true", default=False)

    # Exploratory analysis plots
    parser.add_argument("-expan", "--exploratoryAnalysis", action="store_true", default=False)

    # Classification model 
    parser.add_argument("-dect", "--decisionTree", action="store_true", default=False)
    parser.add_argument("-knn", "--kneighbors", action="store_true", default=False)
    parser.add_argument("-xgb", "--xgboost", action="store_true", default=False)
    parser.add_argument("-randf", "--randomforest", action="store_true", default=False)
    parser.add_argument("-nv", "--naivebayes", action="store_true", default=False)
    parser.add_argument("-strdum", "--stratifieddummyclf", action="store_true", default=False)
    parser.add_argument("-mfdum", "--mostfrequentdummyclf", action="store_true", default=False)
    parser.add_argument("-mlp", "--mlpClassifier", action="store_true", default=False)

    # Autoencoder
    parser.add_argument("-ae", "--autoencoder", action="store_true", default=False)
    parser.add_argument("-fvae", "--fullVectorAutoencoder", action="store_true", default=False)
    parser.add_argument("-useae", "--useAutoencoder", action="store_true", default=False)

    # Features
    parser.add_argument("-gfeat", "--graphFeatures", action="store_true", default=False)
    parser.add_argument("-featv", "--featurevectors", action="store_true", default=False)
    parser.add_argument("-sdfeat", "--selectFeatsBySD", action="store_true", default=False)
    parser.add_argument("-expFeats", "--exportSelectedFeats", action="store_true", default=False)
    parser.add_argument("-expImpMat", "--exportImputatedMatrix", action="store_true", default=False)

    # Labels
    parser.add_argument("-cls", "--classes", action="store_true", default=False)
    parser.add_argument("-tums", "--tumorStage", action="store_true", default=False)
    
    # Graph generation parameters
    parser.add_argument("-e", "--edgeThreshold", type=float, default=0.3)
    parser.add_argument("-rfat", "--runForAllThresholds", action="store_true", default=False)
    #parser.add_argument("-d", "--minDivergenceToKeep", type=float, default=6)

    # Model building parameters
    parser.add_argument("-n", "--numberOfInstances", type=int, default=-1)

    # Multithreading: NOT suggested for now
    global THREADS_TO_USE
    parser.add_argument("-t", "--numberOfThreads", type=int, default=THREADS_TO_USE)

    # Statistical test
    parser.add_argument("-savres", "--saveResults", action="store_true", default=False)
    parser.add_argument("-wilc", "--wilcoxonTest", action="store_true", default=False)
    parser.add_argument("-rwr", "--resetWilcoxonResults", action="store_true", default=False)

    args = parser.parse_args(argv)

    message("Run setup: " + (str(args)))

    # Update global prefix variable
    global Prefix
    Prefix = args.prefixForIntermediateFiles

    # Update global threads to use
    THREADS_TO_USE = args.numberOfThreads

    if args.runForAllThresholds:
        for threshold in np.arange(0.3, 0.85, 0.1):
            threshold = round(threshold, 1)
            # main function
            gMainGraph, mFeatures_noNaNs, vClass, saRemainingFeatureNames, sampleIDs, feat_names, vtumorStage = getGraphAndData(bResetGraph=args.resetGraph,
                                                                                            dEdgeThreshold=threshold,
                                                                                            bResetFiles=args.resetCSVCacheFiles,
                                                                                            bPostProcessing=args.postProcessing,
                                                                                            bstdevFiltering=args.stdevFiltering,
                                                                                            bNormalize=args.normalization,
                                                                                            bNormalizeLog2Scale=args.logScale,
                                                                                            bShow = args.showGraphs, bSave = args.saveGraphs, 
                                                                                            stdevFeatSelection = args.selectFeatsBySD,
                                                                                            nfeat=args.numberOfFeaturesPerLevel, 
                                                                                            expSelectedFeats=args.exportSelectedFeats,
                                                                                            bExportImpMat=args.exportImputatedMatrix)
            #DEBUG LINES 
            print(sampleIDs)
            ################

            # TODO: Restore to NOT reset features
            
            if args.graphFeatures:
                mGraphFeatures = getSampleGraphVectors(gMainGraph, mFeatures_noNaNs, saRemainingFeatureNames, sampleIDs, feat_names,
                                            bResetFeatures=args.resetFeatures, dEdgeThreshold=threshold, 
                                            nfeat=args.numberOfFeaturesPerLevel, bShowGraphs=args.showGraphs, 
                                            bSaveGraphs=args.saveGraphs, stdevFeatSelection = args.selectFeatsBySD)
                
            #DEBUG LINES
            message("mGraphFeatures: ")
            message(mGraphFeatures)
            ##############

    else:
    # main function
        gMainGraph, mFeatures_noNaNs, vClass, saRemainingFeatureNames, sampleIDs, feat_names, vtumorStage = getGraphAndData(bResetGraph=args.resetGraph,
                                                                                        dEdgeThreshold=args.edgeThreshold,
                                                                                        bResetFiles=args.resetCSVCacheFiles,
                                                                                        bPostProcessing=args.postProcessing,
                                                                                        bstdevFiltering=args.stdevFiltering,
                                                                                        bNormalize=args.normalization,
                                                                                        bNormalizeLog2Scale=args.logScale,
                                                                                        bShow = args.showGraphs, bSave = args.saveGraphs, 
                                                                                        stdevFeatSelection = args.selectFeatsBySD,
                                                                                        nfeat=args.numberOfFeaturesPerLevel, 
                                                                                        expSelectedFeats=args.exportSelectedFeats,
                                                                                        bExportImpMat=args.exportImputatedMatrix)
        #DEBUG LINES 
        print(sampleIDs)
        ################

        # TODO: Restore to NOT reset features 
        
        if args.graphFeatures:
            mGraphFeatures = getSampleGraphVectors(gMainGraph, mFeatures_noNaNs, saRemainingFeatureNames, sampleIDs, feat_names,
                                            bResetFeatures=args.resetFeatures, dEdgeThreshold=args.edgeThreshold, 
                                            nfeat=args.numberOfFeaturesPerLevel, bShowGraphs=args.showGraphs, 
                                            bSaveGraphs=args.saveGraphs, stdevFeatSelection = args.selectFeatsBySD)
            
            #DEBUG LINES
            message("mGraphFeatures: ")
            message(mGraphFeatures)
            ##############
        
        #DEBUG LINES 
        # print("sampleIDs: ")
        # print(sampleIDs)
        ################
    
    if args.exploratoryAnalysis:
        plotDistributions(mFeatures_noNaNs, feat_names, stdfeat=args.stdevFiltering, preprocessing=args.postProcessing)

        #plotSDdistributions(mFeatures_noNaNs, feat_names)
        
        #mGraphDistribution(mFeatures_noNaNs, feat_names, startThreshold = 0.3, endThreshold = 0.8, nfeat=args.numberOfFeaturesPerLevel, bResetGraph=True, stdevFeatSelection = args.selectFeatsBySD)

        #plotExplainedVariance(mFeatures_noNaNs, n_components=100, featSelection = args.stdevFiltering)

    # vGraphFeatures = getGraphVector(gMainGraph)
    # print ("Graph feature vector: %s"%(str(vGraphFeatures)))
    
    # Select a sample
    # mSample = mFeatures_noNaNs[1]
    # vGraphFeatures = getSampleGraphFeatureVector(gMainGraph, mSample, saRemainingFeatureNames)
    # print ("Final graph feature vector: %s"%(str(vGraphFeatures)))

    
        # if args.tumorStage:
        #     filteredFeatures, filteredGraphFeatures, filteredTumorStage = filterTumorStage(mFeatures_noNaNs, mGraphFeatures, vSelectedtumorStage, vClass, sampleIDs)
        #     if args.scalingDeactivation:
        #         filteredGraphFeatures = graphVectorPreprocessing(filteredGraphFeatures)

        # if args.scalingDeactivation:
        #     mGraphFeatures = graphVectorPreprocessing(mGraphFeatures)
        

        # #DEBUG LINES
        # message("Max per column: " + str(mGraphFeatures.max(axis=0)))
        # message("Min per column: " + str(mGraphFeatures.min(axis=0)))
        # message(mGraphFeatures)
        # ##################

        # Get selected instance classes
    if args.autoencoder:
        aencoder(mFeatures_noNaNs)

    
    if args.useAutoencoder:
        mFeatures_noNaNs = useAencoder(mFeatures_noNaNs)
        #DEBUG LINES
        message("Matrix shape after autoencoder: " + str(np.shape(mFeatures_noNaNs)))
        #########

    if args.numberOfInstances < 0:
        vSelectedSamplesClasses = vClass
        vSelectedtumorStage = vtumorStage
    else:
        vSelectedSamplesClasses = np.concatenate((vClass[0:int(args.numberOfInstances / 2)][:], vClass[-int(args.numberOfInstances / 2):][:]), axis=0)
        vSelectedtumorStage = np.concatenate((vtumorStage[0:int(args.numberOfInstances / 2)][:], vtumorStage[-int(args.numberOfInstances / 2):][:]), axis=0)
    
    if args.graphFeatures:
        filteredFeatures, filteredGraphFeatures, filteredTumorStage, selectedvClass = filterTumorStage(mFeatures_noNaNs, vSelectedtumorStage, vClass, sampleIDs, mGraphFeatures, useGraphFeatures=args.graphFeatures)
    if args.graphFeatures and args.featurevectors:
        filteredFeatures, filteredGraphFeatures, filteredTumorStage, selectedvClass = filterTumorStage(mFeatures_noNaNs, vSelectedtumorStage, vClass, sampleIDs, mGraphFeatures, useGraphFeatures=args.graphFeatures)
    elif args.featurevectors:
        filteredFeatures, filteredTumorStage, selectedvClass = filterTumorStage(mFeatures_noNaNs, vSelectedtumorStage, vClass, sampleIDs, useGraphFeatures=args.graphFeatures)


    if args.tumorStage and args.scalingDeactivation and args.graphFeatures:
        filteredGraphFeatures = graphVectorPreprocessing(filteredGraphFeatures)
        #DEBUG LINES
        message("Graph features for tumor stage with scaling")
        message("Max per column: " + str(filteredGraphFeatures.max(axis=0)))
        message("Min per column: " + str(filteredGraphFeatures.min(axis=0)))
        message(filteredGraphFeatures)
        ##################

    if args.classes and args.scalingDeactivation and args.graphFeatures:
        mGraphFeatures = graphVectorPreprocessing(mGraphFeatures)
        #DEBUG LINES
        message("Graph features for classes with scaling")
        message("Max per column: " + str(mGraphFeatures.max(axis=0)))
        message("Min per column: " + str(mGraphFeatures.min(axis=0)))
        message(mGraphFeatures)
        ##################

    if args.graphFeatures and not args.scalingDeactivation and args.classes:
        message("Graph features for classes without scaling")
        #DEBUG LINES
        message("First sample before filtering: " + str(mGraphFeatures[0, :]))
        ##############
        # Identify columns where all values are the same
        columns_to_keep = ~np.all(mGraphFeatures == mGraphFeatures[0, :], axis=0)

        # Remove columns with the same value
        mGraphFeatures = mGraphFeatures[:, columns_to_keep]

        #DEBUG LINES
        message("First sample after filtering: " + str(mGraphFeatures[0, :]))
        message("Shape of matrix: " + str(np.shape(mGraphFeatures)))
        ##############

    if args.graphFeatures and not args.scalingDeactivation and args.tumorStage:
        message("Graph features for tumor stage without scaling")
        #DEBUG LINES
        message("First sample before filtering: " + str(filteredGraphFeatures[0, :]))
        ##############
        # Identify columns where all values are the same
        columns_to_keep = ~np.all(filteredGraphFeatures == filteredGraphFeatures[0, :], axis=0)

        # Remove columns with the same value
        filteredGraphFeatures = filteredGraphFeatures[:, columns_to_keep]

        #DEBUG LINES
        message("First sample after filtering: " + str(filteredGraphFeatures[0, :]))
        message("Shape of matrix: " + str(np.shape(filteredGraphFeatures)))
        ##############

    if args.graphFeatures:
        #DEBUG LINES
        message("Class") 
        message("First sample after filtering: " + str(mGraphFeatures[0, :]))
        message("Shape of matrix: " + str(np.shape(mGraphFeatures)))
        message("Tumor stage") 
        message("First sample after filtering: " + str(filteredGraphFeatures[0, :]))
        message("Shape of matrix: " + str(np.shape(filteredGraphFeatures)))
        ##############

    metricResults =[]
    savedResults = {}

    # if args.resetWilcoxonResults:
    #     savedResults = {}
    #     message("Loading results for wilcoxon...Failed.")
    
    # else:   
    #     if os.path.exists("wilcoxon_results.pkl"):
    #         # Load file with the F1-macro results
    #         with open("wilcoxon_results.pkl", 'rb') as f:
    #             savedResults = pickle.load(f)
    #             message("Loading results for wilcoxon...Done.")
    #     else:
    #         savedResults = {}
    #         message("Loading results for wilcoxon...Failed.")
    

    if args.graphFeatures:
        graphLabels = ''
        if args.scalingDeactivation:
            graphLabels += '_Scaling'
        if not args.selectFeatsBySD and not args.selectFeatsBySD:
            graphLabels += '_degs'
        
        graphLabels += '_' + str(args.edgeThreshold) + '_' + str(args.numberOfFeaturesPerLevel)

    
    if args.classes and args.graphFeatures:
        # Extract class vector for colors
        aCategories, y = np.unique(vSelectedSamplesClasses, return_inverse=True)
        
        if args.decisionTree:
            message("Decision tree on graph feature vectors and classes")
            classify(mGraphFeatures, y, metricResults, "DT_GFeatures_Class" + graphLabels, savedResults)
        
        if args.kneighbors:
            message("KNN on graph feature vectors and classes")
            kneighbors(mGraphFeatures, y, metricResults, "kNN_GFeatures_Class" + graphLabels, savedResults)

        if args.xgboost:
            message("XGBoost on graph feature vectors and classes")
            xgboost(mGraphFeatures, y, metricResults, "XGB_GFeatures_Class" + graphLabels, savedResults)

        if args.randomforest:
            message("Random Forest on graph feature vectors and classes")
            RandomForest(mGraphFeatures, y, metricResults, "RF_GFeatures_Class" + graphLabels, savedResults)

        if args.naivebayes:
            message("Naive Bayes on graph feature vectors and classes")
            NBayes(mGraphFeatures, y, metricResults, "NV_GFeatures_Class" + graphLabels, savedResults)

        if args.stratifieddummyclf: 
            message("Stratified Dummy Classifier on graph feature vectors and classes")
            stratifiedDummyClf(mGraphFeatures, y, metricResults, "StratDummy_GFeatures_Class" + graphLabels, savedResults) 
        
        if args.mostfrequentdummyclf:
            message("Most frequent Dummy Classifier on graph feature vectors and classes")
            mostFrequentDummyClf(mGraphFeatures, y, metricResults, "MFDummy_GFeatures_Class" + graphLabels, savedResults)
        
        if args.mlpClassifier:
            message("MLP Classifier on graph feature vectors and classes")
            mlpClassifier(mGraphFeatures, y, metricResults, "MLP_GFeatures_Class" + graphLabels, savedResults)


    if args.classes and args.featurevectors:
        # Extract class vector for colors
        aCategories, y = np.unique(vSelectedSamplesClasses, return_inverse=True)
        X, pca3D = getPCA(mFeatures_noNaNs, 100)
        fig = draw3DPCA(X, pca3D, c=y)

        fig.savefig(Prefix + "SelectedSamplesGraphFeaturePCA.pdf")

        if args.selectFeatsBySD or args.stdevFiltering:
            label = '_featureSelection'
        else:
            label = ''
        if args.decisionTree:
            message("Decision tree on feature vectors and classes")
            classify(X, y, metricResults, "DT_FeatureV_Class" + label, savedResults)

        if args.kneighbors:
            message("KNN on feature vectors and classes")
            kneighbors(X, y, metricResults, "kNN_FeatureV_Class" + label, savedResults)

        if args.xgboost:
            message("XGBoost on feature vectors and classes")
            xgboost(X, y, metricResults, "XGB_FeatureV_Class" + label, savedResults)

        if args.randomforest:
            message("Random Forest on feature vectors and classes")
            RandomForest(X, y, metricResults, "RF_FeatureV_Class" + label, savedResults)

        if args.naivebayes:
            message("Naive Bayes on feature vectors and classes")
            NBayes(X, y, metricResults, "NV_FeatureV_Class" + label, savedResults)

        if args.stratifieddummyclf:  
            message("Stratified Dummy Classifier on feature vectors and classes")
            stratifiedDummyClf(X, y, metricResults, "StratDummy_FeatureV_Class" + label, savedResults)

        if args.mostfrequentdummyclf:
            message("Most frequent Dummy Classifier on feature vectors and classes")
            mostFrequentDummyClf(X, y, metricResults, "MFDummy_FeatureV_Class" + label, savedResults)
        
        if args.mlpClassifier:
            message("MLP Classifier on feature vectors and classes")
            mlpClassifier(X, y, metricResults, "MLP_FeatureV_Class" + label, savedResults)

    if args.tumorStage and args.graphFeatures:
        # Extract tumor stages vector for colors
        aCategories, y = np.unique(filteredTumorStage, return_inverse=True)
        if args.decisionTree:
            message("Decision tree on graph feature vectors and tumor stages")
            classify(filteredGraphFeatures, y, metricResults, "DT_GFeatures_TumorStage" + graphLabels, savedResults)
        
        if args.kneighbors:
            message("KNN on graph feature vectors and tumor stages")
            kneighbors(filteredGraphFeatures, y, metricResults, "kNN_GFeatures_TumorStage" + graphLabels, savedResults)

        if args.xgboost:
            message("XGBoost on graph feature vectors and tumor stages")
            xgboost(filteredGraphFeatures, y, metricResults, "XGB_GFeatures_TumorStage" + graphLabels, savedResults)

        if args.randomforest:
            message("Random Forest on graph feature vectors and tumor stages")
            RandomForest(filteredGraphFeatures, y, metricResults, "RF_GFeatures_TumorStage" + graphLabels, savedResults)
        
        if args.naivebayes:
            message("Naive Bayes on graph feature vectors and tumor stages")
            NBayes(filteredGraphFeatures, y, metricResults, "NV_GFeatures_TumorStage" + graphLabels, savedResults)

        if args.stratifieddummyclf:  
            message("Stratified Dummy Classifier on graph feature vectors and tumor stages")
            stratifiedDummyClf(filteredGraphFeatures, y, metricResults, "StratDummy_GFeatures_TumorStage" + graphLabels, savedResults)

        if args.mostfrequentdummyclf:
            message("Most frequent Dummy Classifier on graph feature vectors and tumor stages")
            mostFrequentDummyClf(filteredGraphFeatures, y, metricResults, "MFDummy_GFeatures_TumorStage" + graphLabels, savedResults)
        
        if args.mlpClassifier:
            message("MLP Classifier on graph feature vectors and tumor stages")
            mlpClassifier(filteredGraphFeatures, y, metricResults, "MLP_GFeatures_TumorStage" + graphLabels, savedResults)

        
    if args.tumorStage and args.featurevectors:
        # Extract tumor stages vector for colors
        aCategories, y = np.unique(filteredTumorStage, return_inverse=True)
        X, pca3D = getPCA(filteredFeatures, 100)
        fig = draw3DPCA(X, pca3D, c=y)

        fig.savefig(Prefix + "SelectedSamplesGraphFeaturePCA.pdf")

        if args.selectFeatsBySD or args.stdevFiltering:
            label = '_featureSelection'
        else:
            label = ''

        if args.decisionTree:
            message("Decision tree on feature vectors and tumor stages")
            classify(X, y, metricResults, "DT_FeatureV_TumorStage" + label, savedResults)

        if args.kneighbors:
            message("KNN on feature vectors and tumor stages")
            kneighbors(X, y, metricResults, "kNN_FeatureV_TumorStage" + label, savedResults)

        if args.xgboost:
            message("XGBoost on feature vectors and tumor stages")
            xgboost(X, y, metricResults, "XGB_FeatureV_TumorStage" + label, savedResults)

        if args.randomforest:
            message("Random Forest on feature vectors and tumor stages")
            RandomForest(X, y, metricResults, "RF_FeatureV_TumorStage" + label, savedResults)

        if args.naivebayes:
            message("Naive Bayes on feature vectors and tumor stages")
            NBayes(X, y, metricResults, "NV_FeatureV_TumorStage" + label, savedResults)  

        if args.stratifieddummyclf:  
            message("Stratified Dummy Classifier on feature vectors and tumor stages")
            stratifiedDummyClf(X, y, metricResults, "StratDummy_FeatureV_TumorStage" + label, savedResults)
        
        if args.mostfrequentdummyclf:
            message("Most frequent Dummy Classifier on feature vectors and tumor stages")
            mostFrequentDummyClf(X, y, metricResults, "MFDummy_FeatureV_TumorStage" + label, savedResults)
    
        if args.mlpClassifier:
            message("MLP Classifier on feature vectors and tumor stages")
            mlpClassifier(X, y, metricResults, "MLP_FeatureV_TumorStage" + label, savedResults)
    
    
    if args.saveResults:
        # Convert the nested dictionary to a DataFrame
        new_df = pd.DataFrame.from_dict(savedResults, orient='index').reset_index()
        new_df.rename(columns={'index': 'sfeatClass'}, inplace=True)

        if os.path.exists("saved_results.csv"):
            # Read the existing CSV file into a DataFrame
            existing_df = pd.read_csv("saved_results.csv")
        else:
            # Create an empty DataFrame if the CSV file does not exist
            existing_df = pd.DataFrame()

        # Append the new results to the existing DataFrame
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        # Write the updated DataFrame back to the CSV file
        combined_df.to_csv("saved_results.csv", index=False)

    if args.wilcoxonTest:
        wilcoxonTests(savedResults)

    # end of main function
    metricsdf = pd.DataFrame(metricResults, columns=['Method', 'Mean_Accuracy', "SEM_Accuracy", 'Mean_F1_micro', "SEM_F1_micro", 'Mean_F1_macro', "SEM_F1_macro"])
    
    if len(metricResults) > 1:
        plotAccuracy(metricsdf)
        plotF1micro(metricsdf)
        plotF1macro(metricsdf)


# test()
if __name__ == "__main__":
    main(sys.argv[1:])

#
#
# def ClassifyInstancesToControlAndTumor():
#     pass
#
#
# ClassifyInstancesToControlAndTumor()
#
# def RepresentSampleAsPearsonCorrelations():
#     # extract mean profile of cancer samples
#     # extract mean profile of control samples
#
#     # every instance is represented based on two features:
#     # <pearson correlation of sample "base" feature vector to mean cancer profile,
#     #  pearson correlation of sample "base" feature vector to mean control profile>
#
# def RepresentSampleAsGraphFeatureVector():
#     pass
#
# def RepresentDataAsGraph():
#     pass
#     # For each DNAmeth feature
#         # Connect high values to high miRNA, mRNA values
#         # Connect high values to low miRNA, mRNA values
#
#
# RepresentDataAsGraph()
