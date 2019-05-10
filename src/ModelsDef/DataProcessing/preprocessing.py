import random
import numpy as np
import h5py
import pickle
import re
import math
import time
from sklearn.model_selection import train_test_split


# This is our Protein class
class Protein:

    def __init__(self, name):
        self.name = name
        self.resolution = -1.0
        self.decoys = []

    # Adds a decoy to our decoy list
    def addDecoy(self, decoy):
        self.decoys.append(decoy)

    # Returns our decoy list
    def getDecoys(self):
        return self.decoys

    # Returns the atoms list of a specified decoy inside our decoy list
    def getAtoms(self, decoy):
        # Magical line to get our index using decoy names
        # In case of duplicate name, it returns the first index found
        index = next((index for (index, d) in enumerate(self.decoys) if d["decoy"] == decoy), None)

        # Check if decoy exists
        if index == None:
            raise Exception("DECOY NOT FOUND IN PROTEIN")

        return self.decoys[index]["atoms"]

    # Using our getAtoms function, we find a decoy and append atoms to the decoy's atoms list
    def addAtom(self, decoy, atom):
        self.getAtoms(decoy).append(atom)

    # Sets a resolution for the protein.
    def setResolution(self, resolution):
        self.resolution = resolution

    # Returns the resolution
    def getResolution(self):
        return self.resolution

    # Work In Progress. Eventually, this would create a (120,120,120,11) tensor and it's GDT_TS
    def getData(self, score="gdt_ts", width=20, height=20, depth=20, layers=12):
        coords = []
        scores = []
        for decoy in self.decoys:
            scores.append(decoy[score])
            xyz = np.zeros((width, height, depth, layers))
            for atom in decoy["atoms"][0]:
                #TODO Fix placement
                xyz[int((atom["coordinates"][0])%xyz.shape[0]), int((atom["coordinates"][1])%xyz.shape[1]), int((atom["coordinates"][2])%xyz.shape[0]), self.get_layer(atom)] = self.get_density(atom)
            coords.append(xyz)
        return coords, scores


    def get_layer(self, atom):
        #HARD COPY FROM PAPER REPO
        if atom["chain_type"] == "O":
            if atom["terminal"]:
                return 8
            else:
                return 6
        elif atom["chain_type"] == "OXT" and atom["terminal"]:
            return 8
        elif atom["chain_type"] == "OT2" and atom["terminal"]:
            return 8
        elif atom["chain_type"] == "N":
            return 2
        elif atom["chain_type"] == "C":
            return 9
        elif atom["chain_type"] == "CA":
            return 11
        else:
            fullName = atom["residue"] + atom["chain_type"]

            if fullName == "CYSSG" or fullName == "METSD" or fullName == "MSESE":
                return 1
            elif fullName == "ASNND2" or fullName == "GLNNE2":
                return 2
            elif fullName == "HISND1" or fullName == "HISND2" or fullName == "TRPNE1":
                return 3
            elif fullName == "ARGNH1" or fullName == "ARGNH2" or fullName == "ARGNE":
                return 4
            elif fullName == "LYSNZ":
                return 5
            elif fullName == "ACEO" or fullName == "ASDNOD1" or fullName == "GLNOE1":
                return 6
            elif fullName == "SEROG" or fullName == "THROG1" or fullName == "TYROH":
                return 7
            elif fullName == "ASPOD1" or fullName == "ASPOD2" or fullName == "GLUOE1" or fullName == "GLUOE2":
                return 8
            elif fullName == "ARGCZ" or fullName == "ASPCG" or fullName == "GLUCD" or fullName == "ACEC" or fullName == "ASNCG" or fullName == "GLNCD":
                return 9
            elif fullName == "HISCD2" or fullName == "HISCE1" or fullName == "HISCG" or fullName == "PHECD1" or fullName == "PHECD2" or fullName == "PHECE1" or fullName == "PHECE2" or fullName == "PHECG" or fullName == "PHECZ" or fullName == "TRPCD1" or fullName == "TRPCD2" or fullName == "TRPCE2" or fullName == "TRPCE3" or fullName == "TRPCG" or fullName == "TRPCH2" or fullName == "TRPCZ2" or fullName == "TRPCZ3" or fullName == "TYRCD1" or fullName == "TYRCD2" or fullName == "TYRCE1" or fullName == "TYRCE2" or fullName == "TYRCG" or fullName == "TYRCZ":
                return 10
            elif fullName == "ALACB" or fullName == "ARGCB" or fullName == "ARGCG" or fullName == "ARGCD" or fullName == "ASNCB" or fullName == "ASPCB" or fullName == "GLNCB" or fullName == "GLNCG" or fullName == "GLUCB" or fullName == "GLUCG" or fullName == "HISCB" or fullName == "ILECB" or fullName == "ILECD1" or fullName == "ILECG1" or fullName == "ILECG2" or fullName == "LEUCB" or fullName == "LEUCD1" or fullName == "LEUCD2" or fullName == "LEUCG" or fullName == "LYSCB" or fullName == "LYSCD" or fullName == "LYSCG" or fullName == "LYSCE" or fullName == "METCB" or fullName == "METCE" or fullName == "METCG" or fullName == "MSECB" or fullName == "MSECE" or fullName == "MSECG" or fullName == "PHECB" or fullName == "PROCB" or fullName == "PROCG" or fullName == "PROCD" or fullName == "SERCB" or fullName == "THRCG2" or fullName == "TYRCB" or fullName == "VALCB" or fullName == "VALCG1" or fullName == "VALCG2" or fullName == "ACECH3" or fullName == "THRCB" or fullName == "CYSCB":
                return 11
            else:
                return 0
            #print("Invalid atom: " + fullName)
            #raise Exception("LINE DATA ERROR")


    def get_density(self, atom):
        van_der_waal_radii = {'H' : 1.2, 'C' : 1.7, 'N' : 1.55, 'O' : 1.52,
        'S' : 1.8, 'D' : 1.2, 'F' : 1.47, 'CL' : 1.75, 'BR' : 1.85, 'P' : 1.8,
        'I' : 1.98, 'E' : 1.0, 'X':1.0 , '': 0.0}
        # Source:https://physlab.lums.edu.pk/images/f/f6/Franck_ref2.pdf

        if atom["chain_type_single"] not in van_der_waal_radii:
            print("Invalid atom: " + atom["chain_type_single"])
            raise Exception("LINE DATA ERROR")

        return pow(math.e, -1 * pow(van_der_waal_radii[atom["chain_type_single"]], 2)/2)


def sortDecoysToBins(decoys, scores, Nb=9, scoreType="gdt_ts"):
    maxScore = max(scores)
    minScore = min(scores)

    #Initialize list of empty lists equal to the number of bins Nb
    #scoreBins is used for [] checks and forEach loops since it is a much lighter list content-wise compared to decoyBins
    decoyBins = [[] for _ in range(Nb)]
    scoreBins = [[] for _ in range(Nb)]
    
    #Sort decoys into bins
    for index, score in enumerate(scores):
        #binIndex is calculated based on score type
        if scoreType == "gdt_ts":
            binIndex = math.floor((Nb-1) * (score - minScore) / (maxScore - minScore))
        else:
            binIndex = math.floor((Nb-1) * score)

        scoreBins[binIndex].append(score)
        decoyBins[binIndex].append(decoys[index])


    #Randomly populate empty bins
    #While loop is used to circumvent problem with only moving decoys from unmodified bins
    while [] in scoreBins:
        modifiedBinsIndexes = []
        for index, bin in enumerate(scoreBins):
            if len(bin) == 0:
                modifiedBinsIndexes.append(index)

                #Here we do multipart bin index selection. We don't do bin index selection in one line for readability purposes
                #Here we have  alist of unmodified indexes
                unmodifiedBinsIndexes = [newIndex for newIndex in range(Nb) if newIndex not in modifiedBinsIndexes]

                #If no bins to take values, thus "restart" for loop
                if unmodifiedBinsIndexes == []:
                    break
                
                #Available unmodified bins must have 2 values so we can take one from them
                availableBinsIndexes = [newIndex for newIndex in unmodifiedBinsIndexes if len(scoreBins[newIndex]) > 1]

                if availableBinsIndexes == []:
                    break

                selectedBinIndex = random.choice(availableBinsIndexes)
                
                #RandomDecoy and randomScore are both pairs since we are using the same index
                #We remove by index since removing by element might messup data in case of identical scores
                randomIndexFromSelectedBin = random.choice(range(len(scoreBins[selectedBinIndex])))
                randomDecoy = decoyBins[selectedBinIndex][randomIndexFromSelectedBin]
                randomScore = scoreBins[selectedBinIndex][randomIndexFromSelectedBin]

                decoyBins[index].append(randomDecoy)
                decoyBins[selectedBinIndex].pop(randomIndexFromSelectedBin)

                scoreBins[index].append(randomScore)
                scoreBins[selectedBinIndex].pop(randomIndexFromSelectedBin)

                
    #Shuffle decoys and scores pairwise in all bins
    for decoyBin, scoreBin in zip(decoyBins, scoreBins):
        joinedBins = list(zip(decoyBin, scoreBin))

        random.shuffle(joinedBins)

        decoyBin[:], scoreBin[:] = zip(*joinedBins)

    return decoyBins, scoreBins


def sortAllProteinsToBins(allDecoys, allScores, Nb=9, scoreType="gdt_ts"):
    #In here, proteinBins is a list of decoyBins and scoreBins is a list of scoreBins(from the other function)
    proteinBins = []
    scoreBins = []

    #For every bins returned by our function, we append them to our list here
    for decoys, scores in zip(allDecoys, allScores):
        p, s = sortDecoysToBins(decoys, scores, Nb, scoreType)
        proteinBins.append(p)
        scoreBins.append(s)

    #Now we shuffle the proteins and scores
    #Shuffle decoys and scores pairwise in all bins
    joinedBins = list(zip(proteinBins, scoreBins))

    random.shuffle(joinedBins)

    proteinBins[:], scoreBins[:] = zip(*joinedBins)

    return proteinBins, scoreBins

def saveData(allDecoys, allScores, Nb=9, scoreType="gdt_ts", filename="dataset-V1.hdf5"):
    #This functions just joins all decoys from bin to bin in a one dimensional manner and saves the data

    X = []
    Y = []

    proteinBins, scoreBins = sortAllProteinsToBins(allDecoys, allScores, Nb, scoreType)

    print("Starting 'flattening")

    for p_bin, s_bin in zip(proteinBins, scoreBins):
        maxLength = 0

        for scores in s_bin:
            if len(scores) > maxLength:
                maxLength = len(scores)
        
        for index in range(maxLength):
            for binIndex in range(Nb):
                try:
                    X.append(p_bin[binIndex][index])
                    Y.append(s_bin[binIndex][index])
                except IndexError:
                    continue

    x_train, x_val, y_train, y_val = train_test_split(np.array(X), np.array(Y), test_size=0.30)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.33)

    hf = h5py.File(filename, 'w')
    hf.create_dataset('x_train', data=x_train)
    hf.create_dataset('x_val', data=x_val)
    hf.create_dataset('x_test', data=x_test)
    hf.create_dataset('y_train', data=y_train)
    hf.create_dataset('y_val', data=y_val)
    hf.create_dataset('y_test', data=y_test)
    hf.close()

    print("Data saved:", filename)

    return (x_train, y_train),(x_val,y_val),(x_test,y_test)






if __name__ == "__main__":
    bins = 9
    scoreType = "gdt_ts"

    proteins = pickle.load(open("all_protein_decoy_data.pickle", "rb", -1))
    decoys = []
    scores = []

    for protein in proteins:
        x, y = protein.getData(width=20, height=20, depth=20)
        decoys.append(x)
        scores.append(y)

    print(np.array(x).shape)
    print(np.array(y).shape)

    # decoys = [list(range(20)), list(range(20))] 
    # scores = [[5.6,2.2,5.3,4.1,2.8,1.2,8.1,4.2,77.1,1.1,6.1,8.2,2.1,3.1,4.3,9.0,0.1,6.3,7.1,5.1], [5.6,2.2,5.3,4.1,2.8,1.2,8.1,4.2,77.1,1.1,6.1,8.2,2.1,3.1,4.3,9.0,0.1,6.3,7.1,5.1]]
    saveData(decoys, scores, bins, scoreType, "../data/dataset-V1.hdf5")

    # start_time = time.clock()
    # decoyBins, scoreBins = sortDecoysToBins(x, y, bins)
    # print("--- %s seconds ---" % (time.clock() - start_time))

    # print(np.array(decoyBins).shape)
    # print(np.array(scoreBins).shape)

    