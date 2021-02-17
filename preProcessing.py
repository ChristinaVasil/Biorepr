#!/usr/bin/env python
import argparse
import copy
import itertools
import os
import pickle
import sys
import multiprocessing as mp
# import time
# from multiprocessing import JoinableQueue
# DEBUG LINES
# mp.log_to_stderr(logging.DEBUG)
#############
from queue import Queue
from threading import Thread
from time import time

import graphviz
import matplotlib.pyplot as plt
# WARNING: This line is important for 3d plotting. DO NOT REMOVE
from mpl_toolkits.mplot3d import Axes3D

import networkx as nx
import numpy as np
from networkx import write_multiline_adjlist, read_multiline_adjlist
from networkx.drawing.nx_pydot import pydot_layout
from numpy import random
from scipy.stats import pearsonr
from sklearn import decomposition
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.tree import DecisionTreeClassifier

# Prefix for intermediate files
Prefix = "GG"
THREADS_TO_USE = mp.cpu_count()  # Init to all CPUs


def progress(s):
    sys.stdout.write("%s" % (str(s)))
    sys.stdout.flush()


def message(s):
    sys.stdout.write("%s\n" % (str(s)))
    sys.stdout.flush()


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
    fModalities = open('gdc_sample_sheet.2020-02-19.tsv')
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
    mFeatures_noNaNs, vClass, sampleIDs = initializeFeatureMatrices(False, True)
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
    mFeatures_noNaNs, vClass, sampleIDs = initializeFeatureMatrices(False, True)
    mFeatures_noNaNs = getNonControlFeatureMatrix(mFeatures_noNaNs, vClass)
    message("Opening file... Done.")
    X, pca3DRes = getPCA(mFeatures_noNaNs, 3)

    fig = draw3DPCA(X, pca3DRes)

    fig.savefig("tumorPCA3D.pdf", bbox_inches='tight')


def draw3DPCA(X, pca3DRes, c=None, cmap=plt.cm.gnuplot, spread=False):
    """
    Draw a 3D PCA given, allowing different classes coloring.
    """

    # Percentage of variance explained for each components
    message('explained variance ratio (first 3 components): %s'
            % str(pca3DRes.explained_variance_ratio_))

    if spread:
        X = QuantileTransformer(output_distribution='uniform').fit_transform(X)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], edgecolor='k', c=c, cmap=cmap, depthshade=False)
    ax.set_xlabel("X coordinate (%4.2f)" % (pca3DRes.explained_variance_ratio_[0]))
    ax.set_ylabel("Y coordinate (%4.2f)" % (pca3DRes.explained_variance_ratio_[1]))
    ax.set_zlabel("Z coordinate (%4.2f)" % (pca3DRes.explained_variance_ratio_[2]))
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

def convertTumorType(s):
    """
    Converts tumor types to float numbers, based on an index of classes.

    :param s: The string representing the tumor type.
    :return: A class index mapped to this type.
    """
    fRes = float(["not reported", "stage i", "stage ii", "stage iii", "stage iv", "stage v"].index(s.decode('UTF-8')))
    if int(fRes) == 0:
        return np.nan
    return fRes


def PCAOnAllData():
    """
    Applies and visualizes PCA on all data.
    """
    # Check if we need to reset the files
    bResetFiles = False
    if len(sys.argv) > 1:
        if "-resetFiles" in sys.argv:
            bResetFiles = True

    # Initialize feature matrices
    mFeatures_noNaNs, vClass, sampleIDs = initializeFeatureMatrices(bResetFiles=bResetFiles, bPostProcessing=True)

    message("Applying PCA...")
    X, pca3D = getPCA(mFeatures_noNaNs, 3)

    # Spread
    message("Applying PCA... Done.")

    # Percentage of variance explained for each components
    message('explained variance ratio (first 3 components): %s'
            % str(pca3D.explained_variance_ratio_))

    message('3 components values: %s'
            % str(X))

    # hey there
 # fhekfhel
    message("Plotting PCA graph...")
    # Assign colors
    aCategories, y = np.unique(vClass, return_inverse=True)
    draw3DPCA(X, pca3D, c=y / 2)
    # DEBUG LINES
    message("Returning categories: \n %s" % (str(aCategories)))
    message("Returning categorical vector: \n %s" % (str(y)))
    ############
    # fig = plt.figure()
    # plt.clf()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2],
    #            c=y / 2, cmap=plt.cm.gnuplot, depthshade=False, marker='.')
    #
    # ax.set_xlabel("X coordinate (%4.2f)" % (pca3D.explained_variance_ratio_[0]))
    # ax.set_ylabel("Y coordinate (%4.2f)" % (pca3D.explained_variance_ratio_[1]))
    # # ax.set_zlabel("Z coordinate (%4.2f)"%(pca.explained_variance_ratio_[2]))
    #
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # # ax.set_zticklabels([])
    # plt.show()
    message("Plotting PCA graph... Done.")


def initializeFeatureMatrices(bResetFiles=False, bPostProcessing=True, bNormalize=True,
                              bNormalizeLog2Scale=True):
    """
    Initializes the case/instance feature matrices, also creating intermediate files for faster startup.

    :param bResetFiles: If True, then reset/recalculate intermediate files. Default: False.
    :param bPostProcessing: If True, then apply post-processing to remove NaNs, etc. Default: True.
    :param bNormalize: If True, then apply normalization to the initial data. Default: True.
    :param bNormalizeLog2Scale: If True, then apply log2 scaling after normalization to the initial data. Default: True.
    :return: The initial feature matrix of the cases/instances.
    """
    # Read control
    message("Opening files...")
    # import pandas as pd
    # df = pd.read_csv('./patientAndControlData.csv', sep='\t')
    # df.reindex_axis(sorted(df.columns), axis=1)
    # datafile = df.as_matrix()

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

        # call load_data with allow_pickle implicitly set to true
        datafile = np.load(Prefix + "patientAndControlData.mat.npy")
        labelfile = np.load(Prefix + "patientAndControlDataLabels.mat.npy")

        # restore np.load for future normal usage
        np.load = np_load_old
        ####################

        clinicalfile = loadTumorStage()
        message("Trying to load saved data... Done.")
    except Exception as eCur:
        message("Trying to load saved data... Failed:\n%s" % (str(eCur)))
        message("Trying to load saved data from CSV...")
        fControl = open("./patientAndControlData.csv", "r")
        message("Loading labels and ids...")
        # labelfile, should have stored tumor_stage or labels?
        labelfile = np.genfromtxt(fControl, skip_header=1, usecols=(0, 73662),
                                  missing_values=['NA', "na", '-', '--', 'n/a'], delimiter="\t",
                                  dtype=np.dtype("object")
                                  )

        message("Splitting features, this is the size of labelfile")
        message(np.shape(labelfile))
        # message(labelfile)

        fControl.close()

        message("Loading labels and ids... Done.")
        # # DEBUG LINES
        # message(str(labelfile))
        # #############

        clinicalfile = loadTumorStage()

        datafile = loadPatientAndControlData()
        message("Trying to load saved data from CSV... Done.")

        # Saving
        saveLoadedData(datafile, labelfile)

    message("Opening files... Done.")
    # Split feature set to features/target field
    mFeatures, vClass, sampleIDs = splitFeatures(clinicalfile, datafile, labelfile)


    mControlFeatureMatrix = getControlFeatureMatrix(mFeatures, vClass)
    message("1 .This is the shape of the control matrix:")
    message(np.shape(mControlFeatureMatrix))

    if bPostProcessing:
        mFeatures = postProcessFeatures(mFeatures, mControlFeatureMatrix)

    # Update control matrix, taking into account postprocessed data
    mControlFeatureMatrix = getControlFeatureMatrix(mFeatures, vClass)

    message("2 .This is the shape of the control matrix:")
    message(np.shape(mControlFeatureMatrix))

    if bNormalize:
        mFeatures = normalizeDataByControl(mFeatures, mControlFeatureMatrix, bNormalizeLog2Scale)

    # CV added sampleIDs as return param
    return mFeatures, vClass, sampleIDs
        #, labelfile


def postProcessFeatures(mFeatures, mControlFeatures):
    """
    Post-processes feature matrix to replace NaNs with control instance feature mean values, and also to remove
    all-NaN columns.

    :param mFeatures: The matrix to pre-process.
    :param mControlFeatures: The subset of the input matrix that reflects control instances.
    :return: The post-processed matrix, without NaNs.
    """
    message("Replacing NaNs from feature set...")
    # DEBUG LINES
    message("Data shape before replacement: %s" % (str(np.shape(mFeatures))))
    #############

    # WARNING: Imputer also throws away columns it does not like
    # imputer = Imputer(strategy="mean", missing_values="NaN", verbose=1)
    # mFeatures_noNaNs = imputer.fit_transform(mFeatures)

    # Extract means per control col
    mMeans = np.nanmean(mControlFeatures, axis=0)
    # Find nans
    inds = np.where(np.isnan(mFeatures))
    # Do replacement
    mFeatures[inds] = np.take(mMeans, inds[1])

    message("Are there any NaNs after postProcessing?")
    message(np.any(np.isnan(mFeatures)))
    # DEBUG LINES
    message("Data shape after replacement: %s" % (str(np.shape(mFeatures))))
    #############

    # TODO: Check below
    # WARNING: If a control data feature was fully NaN, but the corresponding case data had only SOME NaN,
    # we would NOT successfully deal with the case data NaN, because there would be no mean to replace them by.

    #############
    message("Replacing NaNs from feature set... Done.")

    # Convert np array to panda dataframe
    arr = np.array(mFeatures)

    message("Removing features that have only NaN values...")
    mask = np.all(np.isnan(mFeatures), axis=0)
    # a = np.array(mFeatures)
    # mask = (np.nan_to_num(a)).any(axis=0)
    # message(mask)

    mFeatures = mFeatures[:, ~mask]
    message("Number of features after removal: %s" % (str(np.shape(mFeatures))))
    message(mFeatures)
    message("Removing features that have only NaN values...Done")

    message("Are there any NaNs after postProcessing?")
    message(np.any(np.isnan(mFeatures)))

    message("This is mFeatures in postProcessing...")
    message(mFeatures)
    return mFeatures

# TODO add sampleid in splitFeatures

#Test

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
    message(np.size(datafile, 1))
    message("This is the label file:")
    message(labelfile)
    message("This is the shape of the labelfile: %s" % (str(np.shape(labelfile))))
    mFeaturesOnly = datafile[:, 1:73662]
    # Create matrix with extra column (to add tumor stage)
    iFeatCount = np.shape(mFeaturesOnly)[1] + 1
    mFeatures = np.zeros((np.shape(mFeaturesOnly)[0], iFeatCount))
    mFeatures[:, :-1] = mFeaturesOnly
    mFeatures[:, iFeatCount - 1] = np.nan
    # For every row
    for iCnt in range(np.shape(labelfile)[0]):
        condlist = clinicalfile[:, 0] == labelfile[iCnt, 0]
        # Create a converter
        tumorStageToInt = np.vectorize(convertTumorType)
        choicelist = tumorStageToInt(clinicalfile[:, 1])
        # Update the last feature, by joining on ID
        mFeatures[iCnt, iFeatCount - 1] = np.select(condlist, choicelist)
    vClass = labelfile[:, 1]
    sampleIDs = labelfile[:, 0]
    print("This is the vClass: ")
    print(vClass)
    # DEBUG LINES
    message("Found classes:\n%s" % (str(vClass)))
    message("Found sample IDs:\n%s" % (str(sampleIDs)))
    #############
    # DEBUG LINES
    # message("Found tumor types:\n%s" % (
    #     "\n".join(["%s:%s" % (x, y) for x, y in zip(labelfile[:, 0], mFeatures[:, iFeatCount - 1])])))
    #############
    message("Splitfeatures: This is the mFeatures...")
    message(mFeatures)
    message("Splitting features... Done.")
    return mFeatures, vClass, sampleIDs


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
    fControl = open("./patientAndControlData.csv", "r")
    # fControl = open("/datastore/cvas/output.txt", "r")
    # Q Chris: I need to remember why -3
    datafile = np.genfromtxt(fControl, skip_header=1, usecols=range(1, 73663 - 1),
                             missing_values=['NA', "na", '-', '--', 'n/a'], delimiter="\t",
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
    # Tumor stage
    message("Loading tumor stage...")
    fClinical = open("clinicalAll.tsv", "r")
    # While loading stage, also convert string to integer
    clinicalfile = np.genfromtxt(fClinical, skip_header=1,
                                 usecols=(1, 11),
                                 missing_values=['NA', "na", '-', '--', 'n/a'], delimiter="\t",
                                 dtype=np.dtype("object"),
                                 # converters={11: lambda s : ["stage i", "stage ii", "stage iii", "stage iv", "stage v"].index(s)}
                                 )
    fClinical.close()
    message("Loading tumor stage... Done.")
    message("This is the clinical file...")
    message(clinicalfile)
    message("These are the dimensions of the clinical file")
    message(np.shape(clinicalfile))
    return clinicalfile


def ClusterAllData():
    """
    Creates k-means-based clustering of the control and tumor data, visualizing the results in a PCA-based 3D space.
    """
    # Initialize feature matrices
    mFeatures_noNaNs, vClass, sampleIDs = initializeFeatureMatrices(bResetFiles=False, bPostProcessing=True)

    message("Separating instances per class...")
    # Perform clustering, initializing the clusters with a control and a patient
    # Set starting points
    npaControlFeatures = getControlFeatureMatrix(mFeatures_noNaNs, vClass)
    npaNonControlFeatures = getNonControlFeatureMatrix(mFeatures_noNaNs, vClass)

    npInitialCentroids = np.array([np.nanmedian(npaControlFeatures[:, :], 0),
                                   np.nanmedian(npaNonControlFeatures[:, :], 0)])

    message("Separating instances per class... Done.")

    message("Applying k-means...")
    # Perform clustering
    clusterer = KMeans(2, npInitialCentroids, n_init=1)
    y_pred = clusterer.fit_predict(mFeatures_noNaNs)
    message("Applying k-means... Done.")

    message("Applying PCA for visualization...")
    X, pca3D = getPCA(mFeatures_noNaNs, 3)
    # X = QuantileTransformer(output_distribution='uniform').fit_transform(X)

    message("Applying PCA for visualization... Done.")

    draw3DPCA(X, pca3D, c=y_pred)
    aCategories, y = np.unique(vClass, return_inverse=True)
    draw3DPCA(X, pca3D, c=y)
    # splt = fig.add_subplot(122
    #                        # , projection='3d'
    #                       )
    # # Assign colors
    # aCategories, y = np.unique(vClass, return_inverse=True)
    # splt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.gnuplot, marker='.'
    # # , depthshade=False
    #              )
    # splt.title = "True"

    message("Plotting... Done.")

    # Calculate performance (number/precent of misplaced controls, number/precent of misplaced tumor samples)


# Find only the control samples # Chris
def getControlFeatureMatrix(mAllData, vLabels):
    """
    Gets the features of control samples only.
    :param mAllData: The full matrix of data (control plus tumor data).
    :param vLabels: The matrix of labels per case/instance.
    :return: The subset of the data matrix, reflecting only control cases/instances.
    """
    message("Finding only the control data...")
    choicelist = mAllData
    condlist = isEqualToString(vLabels, 'Solid_Tissue_Normal')
    message("This is the control feature matrix:")
    print(choicelist[condlist])
    message("Data shape: %s" % (str(np.shape(choicelist))))
    message("Finding only the control data...Done")
    return choicelist[condlist]


def isEqualToString(npaVector, sString):
    """
    Compares the string value of a vector to a given string, token by token.
    :param npaVector: The input vector.
    :param sString: The string to compare to.
    :return: True if equal. Otherwise, False.
    """

    #TODO check whether we have to convert to UTF-8
    aRes = np.array([oCur.decode('UTF-8').strip() for oCur in npaVector[:]])
    aRes = np.array([oCur.strip() for oCur in aRes[:]])
    aStr = np.array([sString.strip() for oCur in npaVector[:]])
    return aRes == aStr


def getNonControlFeatureMatrix(mAllData, vLabels):
    """
    Returns the subset of the feature matrix, corresponding to non-control (i.e. tumor) data.
    :param mAllData: The full feature matrix of case/instance data.
    :param vLabels: The label matrix, defining what instance is what type (control/tumor).
    :return: The subset of the feature matrix, corresponding to non-control (i.e. tumor) data
    """
    choicelist = mAllData
    condlist = isEqualToString(vLabels, 'Primary_Tumor')
    return choicelist[condlist]


def normalizeDataByControl(mFeaturesToNormalize, mControlData, logScale=True):
    """
    Calculates relative change per feature, transforming also to a log 2 norm/scale

    :param mFeaturesToNormalize: The matrix of features to normalize.
    :param mControlData: The control data sub-matrix.
    :param logScale: If True, log scaling will occur to the result. Default: True.
    :return: The normalized and - possibly - log scaled version of the input feature matrix.
    """
    message("Normalizing based on control set...")
    centroid = np.nanmean(mControlData[:, :], 0)
    # Using percentile change instead of ratio, to avoid lower bound problems
    # Q1
    mOut = ((mFeaturesToNormalize - centroid) + 10e-8) / (centroid + 10e-8)
    # DEBUG LINES
    message("Data shape before normalization: %s" % (str(np.shape(mFeaturesToNormalize))))
    #############
    if logScale:
        mOut = np.log2(2.0 + mOut)  # Ascertain positive numbers
    # DEBUG LINES
    message("Data shape after normalization: %s" % (str(np.shape(mOut))))
    #############
    message("Normalizing based on control set... Done.")
    return mOut


def testSpreadingActivation():
    """
    A harmless test of graph drawing and spreading activation effect.
    """
    g = nx.Graph()
    g.add_path([1, 2, 3, 4], weight=0.5)
    g.add_path([2, 6], weight=0.2)
    g.add_path([5, 6], weight=0.8)

    for nNode in g.nodes():
        g.nodes[nNode]['weight'] = nNode * 10

    drawGraph(g)

    spreadingActivation(g)
    drawGraph(g)


def getFeatureNames():
    """
    Returns the names of the features of the data matrix, as a list.
    :return: The list of feature names.
    """
    message("Loading feature names...")
    fControl = open("./patientAndControlData.csv", "r")
    # fControl = open("/datastore/cvas/output.txt", "r")
    saNames = fControl.readline().strip().split("\t")
    lFeatureNames = [sName.strip() for sName in saNames]
    fControl.close()
    message("Loading feature names... Done.")

    return lFeatureNames


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
                dNow = time()
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
            g.add_edge(saFeatures[iFirstFeatIdx], saFeatures[iSecondFeatIdx], weight=round(fCurCorr * 100) / 100)

        # Update queue
        qQueue.task_done()


# Is this the step where we make the generalised graph? The output is one Graph?
def getFeatureGraph(mAllData, dEdgeThreshold=0.30, bResetGraph=True, dMinDivergenceToKeep=np.log2(10e5)):
    """
    Returns the overall feature graph, indicating interconnections between features.

    :param mAllData: The matrix containing all case/instance data.
    :param dEdgeThreshold: The threshold of minimum correlation required to keep an edge.
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
        g = read_multiline_adjlist(Prefix + "graphAdjacencyList.txt", create_using=g)
        with open(Prefix + "usefulFeatureNames.pickle", "rb") as fIn:
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

    iFeatureCount = np.shape(mAllData)[1]
    mMeans = np.nanmean(mAllData, 0)  # Ignore nans

    # Q1 Chris: is this the step where we apply the threshold? WHat is the threshold?
    # So, basically keep in vUseful, only the features that their value is greater than dMinDivergenceToKeep
    vUseful = [abs(mMeans[iFieldNum]) - dMinDivergenceToKeep > 0.00 for iFieldNum in range(1, iFeatureCount)]

    saFeatures = getFeatureNames()[1:iFeatureCount]
    saUsefulIndices = [iFieldNum for iFieldNum, _ in enumerate(saFeatures) if vUseful[iFieldNum]]
    saUsefulFeatureNames = [saFeatures[iFieldNum] for iFieldNum in saUsefulIndices]
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
    message("Routing edge calculation for %d possible pairs..." % (iAllPairs))
    lCombinations = itertools.combinations(saUsefulIndices, 2)

    # Create queue and threads
    threads = []
    num_worker_threads = THREADS_TO_USE  # DONE: Use available processors
    qCombination = Queue(1000 * num_worker_threads)

    # t = threading.Thread(target=addEdgeAboveThreshold, args=(i, qCombination,))
    processes = [Thread(target=addEdgeAboveThreshold, args=(i, qCombination,)) for i in range(num_worker_threads)]
    for t in processes:
        t.setDaemon(True)
        t.start()

    # Feed tasks
    iCnt = 1
    dStartTime = time()
    for iFirstFeatIdx, iSecondFeatIdx in lCombinations:
        qCombination.put((iFirstFeatIdx, iSecondFeatIdx, g, mAllData, saFeatures, iFirstFeatIdx, iSecondFeatIdx,
                          iCnt, iAllPairs, dStartTime, dEdgeThreshold))

        # Wait a while if we reached full queue
        if qCombination.full():
            message("So far routed %d tasks. Waiting on worker threads to provide more tasks..." % (iCnt))
            time.sleep(0.05)

        iCnt += 1
    message("Routing edge calculation for %d possible pairs... Done." % (iAllPairs))

    message("Waiting for completion...")
    qCombination.join()

    message("Total time (sec): %4.2f" % (time() - dStartTime))

    message("Creating edges for %d possible pairs... Done." % (iAllPairs))

    message("Extracting graph... Done.")

    message("Removing single nodes... Nodes before removal: %d" % (g.number_of_nodes()))
    toRemove = [curNode for curNode in g.nodes().keys() if len(g[curNode]) == 0]
    while len(toRemove) > 0:
        g.remove_nodes_from(toRemove)
        toRemove = [curNode for curNode in g.nodes().keys() if len(g[curNode]) == 0]
        message("Nodes after removal step: %d" % (g.number_of_nodes()))
    message("Removing single nodes... Done. Nodes after removal: %d" % (g.number_of_nodes()))

    message("Saving graph...")
    write_multiline_adjlist(g, Prefix + "graphAdjacencyList.txt")
    with open(Prefix + "usefulFeatureNames.pickle", "wb") as fOut:
        pickle.dump(saUsefulFeatureNames, fOut)

    message("Saving graph... Done.")

    message("Trying to load graph... Done.")

    return g, saUsefulFeatureNames


def getGraphAndData(bResetGraph=False, dMinDivergenceToKeep=np.log2(10e6), dEdgeThreshold=0.3,
                    bResetFiles=False, bPostProcessing=True, bNormalize=True, bNormalizeLog2Scale=True):
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
    mFeatures_noNaNs, vClass, sampleIDs = initializeFeatureMatrices(bResetFiles=bResetFiles, bPostProcessing=bPostProcessing,
                                                         bNormalize=bNormalize, bNormalizeLog2Scale=bNormalizeLog2Scale)
    gToDraw, saRemainingFeatureNames = getFeatureGraph(mFeatures_noNaNs, dEdgeThreshold=dEdgeThreshold,
                                                       bResetGraph=bResetGraph,
                                                       dMinDivergenceToKeep=dMinDivergenceToKeep)

    return gToDraw, mFeatures_noNaNs, vClass, saRemainingFeatureNames, sampleIDs


def drawGraph(gToDraw, bShow = True):
    """
    Draws and displays a given graph, by using graphviz.

    :param gToDraw: The graph to draw.
    """
    plt.figure(figsize=(len(gToDraw.edges()) * 2, len(gToDraw.edges()) * 2))
    plt.clf()
    # pos = graphviz_layout(gToDraw)
    pos = pydot_layout(gToDraw)
    try:
        dNodeLabels = {}
        # For each node
        for nCurNode in gToDraw.nodes():
            # Try to add weight
            dNodeLabels[nCurNode] = "%s (%4.2f)" % (str(nCurNode), gToDraw.nodes[nCurNode]['weight'])
    except KeyError:
        # Weights could not be added, use nodes as usual
        dNodeLabels = None

    nx.draw_networkx(gToDraw, pos, arrows=False, node_size=1200, color="blue", with_labels=True, labels=dNodeLabels)
    labels = nx.get_edge_attributes(gToDraw, 'weight')
    nx.draw_networkx_edge_labels(gToDraw, pos, edge_labels=labels)

    if bShow:
        plt.show()


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
         # Avg deg centrality
         nx.algorithms.clique.graph_number_of_cliques(gGraph),
         # TODO: Shortest path does NOT work.; revisit if needed
         # nx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path_length(gGraph),
         # getAvgShortestPath(gGraph)
         nx.algorithms.connectivity.connectivity.average_node_connectivity(gGraph),
         np.mean([np.mean(list(x[1].values())) for x in
                  list(nx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path_length(gGraph))]),
         ]
    )
    # DEBUG LINES
    message("Extracting graph feature vector... Done.")

    return mRes


# PCAOnControl()

# PCAOnAllData()

# ClusterAllData()

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
    # In each iteration
    for iIterCnt in range(iIterations):
        # For every node
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


def assignSampleValuesToGraphNodes(gGraph, mSample, saSampleFeatureNames):
    """
    Assigns values/weights to nodes of a given graph (inplace), for a given sample.
    :param gGraph: The generic graph.
    :param mSample: The sample which will define the feature node values/weights.
    :param saSampleFeatureNames: The mapping between feature names and indices.
    """
    # For each node
    for nNode in gGraph.nodes():
        # Get corresponding feature idx in sample
        iFeatIdx = saSampleFeatureNames.index(nNode)
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
    # Find appropriate percentile
    dMinWeight = np.percentile(mWeights, (1.0 - dKeepRatio) * 100)
    # Select and remove nodes with lower value
    toRemove = [curNode for curNode in gMainGraph.nodes().keys() if gMainGraph.nodes[curNode]['weight'] < dMinWeight]
    gMainGraph.remove_nodes_from(toRemove)

    return gMainGraph



def showAndSaveGraph(gToDraw, sPDFFileName="corrGraph.pdf",bShow = True, bSave = True ):
    """
    Draws and displays a given graph, also saving it to a given file.
    :param gToDraw: The graph to draw and save.
    :param sPDFFileName:  The output filename. Default: corrGraph.pdf.
    """
    message("Displaying graph...")
    drawGraph(gToDraw, bShow)
    message("Displaying graph... Done.")

    message("Saving graph to file...")
    if bSave:
        plt.savefig(sPDFFileName, bbox_inches='tight')
    message("Saving graph to file... Done.")


def generateAllSampleGraphFeatureVectors(gMainGraph, mAllSamples, saRemainingFeatureNames, sampleIDs):
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
    for i in range(num_worker_threads):
        t = Thread(target=getSampleGraphFeatureVector, args=(i, qTasks,))
        t.setDaemon(True)
        t.start()

    # Count instances
    iAllCount = np.shape(mAllSamples)[0]

    # Item iterator
    iCnt = iter(range(1, iAllCount + 1))
    dStartTime = time()

    # Init result list
    lResList = []
    # Add all items to queue
    np.apply_along_axis(
        lambda mSample: qTasks.put((sampleIDs, lResList, gMainGraph, mSample, saRemainingFeatureNames, next(iCnt), iAllCount,
                                    dStartTime)), 1, mAllSamples)

    message("Waiting for completion...")
    qTasks.join()
    message("Total time (sec): %4.2f" % (time() - dStartTime))

    return np.array(lResList)

    ########################
    # LINEAR EXECUTION
    ########################
    # # For all samples
    # dStartTime = time()
    # iAllCount = np.shape(mAllSamples)[1] # Get rows/instances
    # iCnt = iter(range(1,iAllCount+1))
    # # Get the sample vector
    # return np.apply_along_axis(
    #     lambda mSample: getSampleGraphFeatureVector(gMainGraph, mSample, saRemainingFeatureNames, next(iCnt), iAllCount, dStartTime), 1, mAllSamples)


def getSampleGraphFeatureVector(i, qQueue):
    """
    Helper parallelization function, which calculates the graph representation of a given sample.
    :param i: The thread number calling the helper.
    :param qQueue: A Queue, from which the execution data will be drawn. Should contain:
    lResList -- reference to the list containing the result
    gMainGraph -- the generic graph of feature correlations
    mSample -- the sample to represent
    saRemainingFeatureNames -- the list of useful feature names
    iCnt -- the current sample count
    iAllCount -- the number of all samples to be represented
    dStartTime -- the time when parallelization started
    """

    # dSample = {}

    while True:
        sampleID, lResList, gMainGraph, mSample, saRemainingFeatureNames, iCnt, iAllCount, dStartTime = qQueue.get()

        # DEBUG LINES
        message("Working on instance %d of %d..." % (iCnt, iAllCount))
        #############

        # Create a copy of the graph
        # gMainGraph = gMainGraph.copy()
        gMainGraph = copy.deepcopy(gMainGraph)

        # Assign values
        assignSampleValuesToGraphNodes(gMainGraph, mSample, saRemainingFeatureNames)
        # Apply spreading activation
        gMainGraph = spreadingActivation(gMainGraph, bAbsoluteMass=True)  # TODO: Add parameter, if needed
        # Keep top performer nodes
        gMainGraph = filterGraphNodes(gMainGraph, dKeepRatio=0.25)  # TODO: Add parameter, if needed
        # Extract and return features
        vGraphFeatures = getGraphVector(gMainGraph)
        # Save graph to .dot arxeio
        # graphviz swse se grapho
        # kalese to draw and showAndsave graph,
        # ftiakse ena katalogo poy na dexetai to sample and to graph

        # All samples dict


        #dSample[str(mSample)] = gMainGraph
        #print(dSample)

        #print("Showing and saving the graph of sample %s" % mSample)
        #param: sample id,
        message("Calling showAndSaveGraph...")
        showAndSaveGraph(gMainGraph, sPDFFileName = "SampleID%s" % sampleID)

        #  Add to common result queue
        lResList.append(vGraphFeatures)

        # Signal done
        qQueue.task_done()

        # DEBUG LINES
        if iCnt % 5 == 0 and (iCnt != 0):
            dNow = time()
            dRate = ((dNow - dStartTime) / iCnt)
            dRemaining = (iAllCount - iCnt) * dRate
            message("%d (Estimated remaining (sec): %4.2f - Working at a rate of %4.2f samples/sec)\n" % (
                iCnt, dRemaining, 1.0 / dRate))
        #############


def classify(X, y):
    """
    Calculates and outputs the performance of classification, through 10-fold cross-valuation, given a set of feature vectors and a set of labels.
    :param X: The feature vector matrix.
    :param y: The labels.
    """
    classifier = DecisionTreeClassifier()
    scores = cross_val_score(classifier, X, y, cv=min(10, len(y)));
    message("Avg. Performanace: %4.2f (st. dev. %4.2f) \n %s" % (np.mean(scores), np.std(scores), str(scores)))

    # Output model
    classifier.fit(X, y)
    dot_data = tree.export_graphviz(classifier, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("Rules")


def getSampleGraphVectors(gMainGraph, mFeatures_noNaNs, saRemainingFeatureNames, sampleIDs, bResetFeatures=True,
                          numOfSelectedSamples=-1):
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
        with open(Prefix + "graphFeatures.pickle", "rb") as fIn:
            mGraphFeatures = pickle.load(fIn)
        message("Trying to load graph feature matrix... Done.")
    except Exception as e:
        message("Trying to load graph feature matrix... Failed:\n%s" % (str(e)))
        message("Computing graph feature matrix...")

        if (numOfSelectedSamples < 0):
            mSamplesSelected = mFeatures_noNaNs
        else:
            mSamplesSelected = np.concatenate((mFeatures_noNaNs[0:int(numOfSelectedSamples / 2)][:],
                                               mFeatures_noNaNs[-int(numOfSelectedSamples / 2):][:]), axis=0)

        message("Extracted selected samples:\n" + str(mSamplesSelected[:][0:10]))
        # Extract vectors
        # TODO pass SampleID to generateAllSampleGraphFeatureVectors
        mGraphFeatures = generateAllSampleGraphFeatureVectors(gMainGraph, mSamplesSelected, saRemainingFeatureNames, sampleIDs)
        message("Computing graph feature matrix... Done.")

        message("Saving graph feature matrix...")
        with open(Prefix + "graphFeatures.pickle", "wb") as fOut:
            pickle.dump(mGraphFeatures, fOut)
        message("Saving graph feature matrix... Done.")
    return mGraphFeatures


def main(argv):
    # Init arguments
    parser = argparse.ArgumentParser(description='Perform bladder tumor analysis experiments.')

    # File caching / intermediate files
    parser.add_argument("-rc", "--resetCSVCacheFiles", action="store_true", default=False)
    parser.add_argument("-rg", "--resetGraph", action="store_true", default=False)
    parser.add_argument("-rf", "--resetFeatures", action="store_true", default=False)
    parser.add_argument("-pre", "--prefixForIntermediateFiles", default="")

    # Post-processing control
    parser.add_argument("-p", "--postProcessing", action="store_true",
                        default=True)  # If False NO postprocessing occurs
    parser.add_argument("-norm", "--normalization", action="store_true", default=True)
    parser.add_argument("-ls", "--logScale", action="store_true", default=True)

    # Graph generation parameters
    parser.add_argument("-e", "--edgeThreshold", type=float, default=0.3)
    parser.add_argument("-d", "--minDivergenceToKeep", type=float, default=6)

    # Model building parameters
    parser.add_argument("-n", "--numberOfInstances", type=int, default=-1)

    # Multithreading: NOT suggested for now
    global THREADS_TO_USE
    parser.add_argument("-t", "--numberOfThreads", type=int, default=THREADS_TO_USE)

    args = parser.parse_args(argv)

    message("Run setup: " + (str(args)))

    # Update global prefix variable
    global Prefix
    Prefix = args.prefixForIntermediateFiles

    # Update global threads to use
    THREADS_TO_USE = args.numberOfThreads

    # # main function
    gMainGraph, mFeatures_noNaNs, vClass, saRemainingFeatureNames, sampleIDs = getGraphAndData(bResetGraph=args.resetGraph,
                                                                                    dEdgeThreshold=args.edgeThreshold,
                                                                                    dMinDivergenceToKeep=args.minDivergenceToKeep,
                                                                                    bResetFiles=args.resetCSVCacheFiles,
                                                                                    bPostProcessing=args.postProcessing,
                                                                                    bNormalize=args.normalization,
                                                                                    bNormalizeLog2Scale=args.logScale)
    # vGraphFeatures = getGraphVector(gMainGraph)
    # print ("Graph feature vector: %s"%(str(vGraphFeatures)))

    # Select a sample
    # mSample = mFeatures_noNaNs[1]
    # vGraphFeatures = getSampleGraphFeatureVector(gMainGraph, mSample, saRemainingFeatureNames)
    # print ("Final graph feature vector: %s"%(str(vGraphFeatures)))

    # TODO: Restore to NOT reset features
    mGraphFeatures = getSampleGraphVectors(gMainGraph, mFeatures_noNaNs, saRemainingFeatureNames, sampleIDs,
                                           bResetFeatures=args.resetFeatures,
                                           numOfSelectedSamples=args.numberOfInstances)

    # Perform PCA
    # Get selected instance classes
    if args.numberOfInstances < 0:
        vSelectedSamplesClasses = vClass
    else:
        vSelectedSamplesClasses = np.concatenate(
            (vClass[0:int(args.numberOfInstances / 2)][:], vClass[-int(args.numberOfInstances / 2):][:]), axis=0)

    # Extract class vector for colors
    aCategories, y = np.unique(vSelectedSamplesClasses, return_inverse=True)
    X, pca3D = getPCA(mGraphFeatures, 3)
    fig = draw3DPCA(X, pca3D, c=y)
    fig.savefig(Prefix + "SelectedSamplesGraphFeaturePCA.pdf")

    classify(X, y)

    # end of main function


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
