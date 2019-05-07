import os
import pickle
import re
import numpy as np
import math


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
                xyz[int((atom["coordinates"][0])%xyz.shape[0]), int((atom["coordinates"][1])%xyz.shape[1]), int((atom["coordinates"][2])%xyz.shape[0]), get_layer(atom)] = get_density(atom)
            coords.append(xyz)
        return coords, scores


def get_layer(atom):
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


def get_density(atom):
    van_der_waal_radii = {'H' : 1.2, 'C' : 1.7, 'N' : 1.55, 'O' : 1.52,
    'S' : 1.8, 'D' : 1.2, 'F' : 1.47, 'CL' : 1.75, 'BR' : 1.85, 'P' : 1.8,
    'I' : 1.98, 'E' : 1.0, 'X':1.0 , '': 0.0}
    # Source:https://physlab.lums.edu.pk/images/f/f6/Franck_ref2.pdf

    if atom["chain_type_single"] not in van_der_waal_radii:
        print("Invalid atom: " + atom["chain_type_single"])
        raise Exception("LINE DATA ERROR")

    return pow(math.e, -1 * pow(van_der_waal_radii[atom["chain_type_single"]], 2)/2)


def readAndStoreData(image_dir):
    # Here we will store a list of Protein instances
    proteins = []

    # Here we iterate throught he root folder of the dataset
    for root, dirs, filenames in os.walk(image_dir):
        # Here we go through every folder in the dataset
        for dir in dirs:
            # Nothing good in the Description folder so we ignore it
            if dir != "Description":
                print("Working on protein: ", dir)
                protein = Protein(dir)

                # Here we iterate trhough every file in current folder
                for f in os.listdir(image_dir + "/" + dir):
                    # List.dat contain scores per decoy and decoy name
                    if (f == "list.dat"):
                        with open(image_dir + "/" + dir + "/" + f) as file:
                            for line in file:
                                # Here we remove all tabs and split curret line into text
                                line = line.split("\t")
                                # Here we remove a new line character
                                line[4] = line[4].replace("\n", "")

                                # Decoys start with server. Don't know why decoys have those names xD
                                # A decoy is a dicionary with various info like scores and a list of atoms
                                if (line[0].startswith("server")):
                                    decoy = {
                                        "decoy": line[0],
                                        "rmsd": float(line[1]),
                                        "tmscore": float(line[2]),
                                        "gdt_ts": float(line[3]),
                                        "gdt_ha": float(line[4]),
                                        "atoms": []
                                    }
                                    # Now we add the decoy dictionary to our protein intance
                                    protein.addDecoy(decoy)

                    # Here we are working with the files that contain server in their names.
                    # These files contain a list of atoms for decoy number X
                    elif (f != dir + ".pdb"):
                        with open(image_dir + "/" + dir + "/" + f) as file:
                            # Here we will store a list of atoms who each is a dictionary
                            atoms = []
                            terminal = True
                            temp = -1
                            for line in file:
                                # Here we split the lines by space and remove set to empty unneded string
                                line = line.split(" ")
                                line[0] = ""  # This is just the string ATOM in every line
                                line[-1] = ""  # This is a newline character

                                # Here we iterate indefinitely until all empty strings are removed
                                while ("" in line):
                                    line.remove("")

                                # Here we fix the monstrosities in the decoy files like
                                # ATOM    660  CD2 LEU    65     124.825 146.253-102.428  1.00                 C
                                # ATOM    660  CD2 LEU    65     146.253-102.428 124.825  1.00                 C
                                # ATOM    660  CD2 LEU    65     124.825-146.253-102.428  1.00                 C
                                # ATOM    660  CD2 LEU    65     -124.825-146.253-102.428  1.00                 C
                                if len(line) == 7:
                                    line.append("")
                                    line.append("")
                                    line[8] = line[6]
                                    line[7] = line[5]

                                    temp = re.split('(-)', line[4])

                                    while ("" in temp):
                                        temp.remove("")

                                    if len(temp) == 6:
                                        line[6] = temp[4] + temp[5]
                                        line[5] = temp[2] + temp[3]
                                        line[4] = temp[0] + temp[1]
                                    elif len(temp) == 5 and temp[1] == "-":
                                        line[6] = temp[3] + temp[4]
                                        line[5] = temp[1] + temp[2]
                                        line[4] = temp[0]
                                    else:
                                        print("There was an error auto correcting data field in line", line[0],
                                              "in protein", dir + "'s decoy", f)
                                        print("This is the line:", line)
                                        raise Exception("WRONG LINE FORMAT-0. CHECK LINE 127 IN preprocessing")
                                elif len(line) == 8:

                                    line.append("")
                                    line[8] = line[7]
                                    line[7] = line[6]

                                    temp = re.split('(-)', line[4])
                                    temp2 = re.split('(-)', line[5])

                                    while ("" in temp):
                                        temp.remove("")
                                    while ("" in temp2):
                                        temp2.remove("")

                                    if len(temp) > len(temp2):
                                        if len(temp) == 3:
                                            line[5] = temp[1] + temp[2]
                                            line[4] = temp[0]
                                        elif len(temp) == 4:
                                            line[5] = temp[2] + temp[3]
                                            line[4] = temp[0] + temp[1]
                                        else:
                                            print("There was an error auto correcting data field in line", line[0],
                                                  "in protein", dir + "'s decoy", f)
                                            print("This is the line:", line)
                                            raise Exception("WRONG LINE FORMAT-1. CHECK LINE 134 IN preprocessing")
                                    elif len(temp) < len(temp2):
                                        if len(temp2) == 3:
                                            line[6] = temp2[1] + temp2[2]
                                            line[5] = temp2[0]
                                        elif len(temp2) == 4:
                                            line[6] = temp2[2] + temp2[3]
                                            line[5] = temp2[0] + temp2[1]
                                        else:
                                            print("There was an error auto correcting data field in line", line[0],
                                                  "in protein", dir + "'s decoy", f)
                                            print("This is the line:", line)
                                            raise Exception("WRONG LINE FORMAT-2. CHECK LINE 145 IN preprocessing")
                                    else:
                                        print("There was an error auto correcting data field in line", line[0],
                                              "in protein", dir + "'s decoy", f)
                                        print("This is the line:", line)
                                        raise Exception("WRONG LINE FORMAT-3. CHECK LINE 149 IN preprocessing")

                                # Here we finished fixing those weird lines. Now continue with data parsing
                                if len(line) == 9:
                                    # Just for debugging strange cases of abnormal atom counts
                                    if (line[7] != "1.00"):
                                        print("Protein", dir + "'s decoy", f + ", has an atom count of", line[7],
                                              "in line", line[0])

                                    try:
                                        # Every atom is a dictionary with it's own data
                                        if temp == -1 or temp != int(line[3]):
                                            temp = int(line[3])
                                            terminal = True
                                        atom = {
                                            "terminal": terminal,
                                            "chain_type": line[1],
                                            "residue": line[2],
                                            "residue_number": int(line[3]),
                                            "coordinates": [float(line[4]), float(line[5]), float(line[6])],
                                            "atom_count": float(line[7]),
                                            "chain_type_single": line[8]
                                        }
                                        terminal = False
                                        # We add the current atom to our atoms list
                                        atoms.append(atom)
                                    except:
                                        print("There was an error with the current line's data", line[0], "in protein",
                                              dir + ", decoy", f)
                                        raise Exception("LINE DATA ERROR")
                                else:
                                    if line != []:
                                        print("Wrong line format", line[0], "in protein", dir + "'s decoy", f)
                                        print("This is the line:", line)

                            # Here we went through all atoms. Now we add those atoms to the current decoy
                            # Remember that f is the name of our file and the name of our decoy too
                            protein.addAtom(f, atoms)

                    else:
                        with open(image_dir + "/" + dir + "/" + f) as file:

                            # For now, we only use the PDB to get data on the protein resolution.
                            resolutionFound = False
                            for line in file:
                                if "RESOLUTION" in line:
                                    resolutionFound = True

                                    # Here we remove everything we don't need from the current line
                                    line = line.split(" ")
                                    while ("" in line):
                                        line.remove("")
                                    if "\n" in line:
                                        line.remove("\n")
                                    if "\n" in line[-1]:
                                        line[-1] = line[-1].replace("\n", "")

                                    # Now we set the Resolution of our protein
                                    protein.setResolution(float(line[-1]))

                            # Not having a resolution will jsut print debugging info to the console
                            if resolutionFound == False:
                                print("Didn't find Resolution for protein:", dir)

                                # We've gone through all data for the current protein. Now we add the protein instance to our protein list
                proteins.append(protein)

    # Since we've finished reading all proteins, now we store them
    with open('all_protein_decoy_data.pickle', 'wb') as f:
        pickle.dump(proteins, f, -1)

def loadData():
    #TODO implement this with dignity >:l
    readAndStoreData("C:\\Users\\Owrn\\Documents\\gitRepos\\3DCNN_MQA_Python\\src\\ModelsDef\\CASP11Stage1_SCWRL")
    proteins = pickle.load(open("all_protein_decoy_data.pickle", "rb", -1))

    return ([proteins[1].getData()[0]], proteins[1].getData()[1]), ([proteins[2].getData()[0]], proteins[2].getData()[1])

if __name__ == "__main__":
    # Here we call our awesome parser :)
    #readAndStoreData("CASP11Stage1_SCWRL")

    # This is to test data loading from the pickle file
    proteins = pickle.load(open("all_protein_decoy_data.pickle", "rb", -1))
    x,y = proteins[0].getData()
    print(np.array(x).shape)
    print(np.array(y).shape)