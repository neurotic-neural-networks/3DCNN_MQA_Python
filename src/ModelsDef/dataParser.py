import os
import random
import cv2
import numpy as np
import h5py
import pickle
import re

#This is our Protein class
class Protein:

    def __init__(self, name):
        self.name = name
        self.resolution = -1.0
        self.decoys = []

    #Adds a decoy to our decoy list
    def addDecoy(self, decoy):
        self.decoys.append(decoy)
    
    #Returns our decoy list
    def getDecoys(self):
        return self.decoys

    #Returns the atoms list of a specified decoy inside our decoy list
    def getAtoms(self, decoy):
        #Magical line to get our index using decoy names
        #In case of duplicate name, it returns the first index found
        index = next((index for (index, d) in enumerate(self.decoys) if d["decoy"] == decoy), None)
        
        #Check if decoy exists
        if index == None:
            raise Exception("DECOY NOT FOUND IN PROTEIN")

        return self.decoys[index]["atoms"]

    #Using our getAtoms function, we find a decoy and append atoms to the decoy's atoms list
    def addAtom(self, decoy, atom):
        self.getAtoms(decoy).append(atom)

    #Sets a resolution for the protein. 
    def setResolution(self, resolution):
        self.resolution = resolution

    #Returns the resolution
    def getResolution(self):
        return self.resolution

    #Work In Progress. Eventually, this would create a (120,120,120,11) tensor and it's GDT_TS
    def getData(self, score = "tmscore"):
        coords = []
        scores = []
        for decoy in self.decoys:
            scores.append(decoy[score])
            xyz = []

            for atom in decoy["atoms"][0]:
                #print(atom["coordinates"])
                xyz.append(atom["coordinates"])
            #print(coordinates)
            coords.append(xyz)
        
        return (coords, scores)


def readAndStoreData(image_dir):
    #Here we will store a list of Protein instances
    proteins = []

    #Here we iterate throught he root folder of the dataset
    for root, dirs, filenames in os.walk(image_dir):
        #Here we go through every folder in the dataset
        for dir in dirs:
            #Nothing good in the Description folder so we ignore it
            if dir != "Description":
                print("Working on protein:", dir)
                protein = Protein(dir)

                #Here we iterate trhough every file in current folder
                for f in os.listdir(image_dir + "/" + dir):
                    #List.dat contain scores per decoy and decoy name
                    if(f == "list.dat"):
                        with open(image_dir + "/" + dir + "/" + f) as file:
                            for line in file:
                                #Here we remove all tabs and split curret line into text
                                line = line.split("\t")
                                #Here we remove a new line character
                                line[4] = line[4].replace("\n", "")

                                #Decoys start with server. Don't know why decoys have those names xD
                                #A decoy is a dicionary with various info like scores and a list of atoms
                                if (line[0].startswith("server")):
                                    decoy = {
                                        "decoy": line[0],
                                        "rmsd": float(line[1]),
                                        "tmscore": float(line[2]),
                                        "gdt_ts": float(line[3]),
                                        "gdt_ha": float(line[4]),
                                        "atoms": []
                                    }
                                    #Now we add the decoy dictionary to our protein intance
                                    protein.addDecoy(decoy)

                    #Here we are working with the files that contain server in their names.
                    #These files contain a list of atoms for decoy number X
                    elif (f != dir + ".pdb"):
                        with open(image_dir + "/" + dir + "/" + f) as file:
                            #Here we will store a list of atoms who each is a dictionary
                            atoms = []
                            for line in file:
                                #Here we split the lines by space and remove set to empty unneded string
                                line = line.split(" ")
                                line[0] = "" #This is just the string ATOM in every line
                                line[-1] = "" #This is a newline character

                                #Here we iterate indefinitely until all empty strings are removed
                                while ("" in line):
                                    line.remove("")

                                
                                #Here we fix the monstrosities in the decoy files like
                                #ATOM    660  CD2 LEU    65     124.825 146.253-102.428  1.00                 C
                                #ATOM    660  CD2 LEU    65     146.253-102.428 124.825  1.00                 C  
                                #ATOM    660  CD2 LEU    65     124.825-146.253-102.428  1.00                 C
                                #ATOM    660  CD2 LEU    65     -124.825-146.253-102.428  1.00                 C  
                                if len(line) == 7:
                                    line.append("")
                                    line.append("")
                                    line[8] = line[6]
                                    line[7] = line[5]

                                    temp = re.split('(-)', line[4])

                                    while("" in temp):
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
                                        print("There was an error auto correcting data field in line", line[0], "in protein", dir +"'s decoy", f)
                                        print("This is the line:", line)
                                        raise Exception("WRONG LINE FORMAT-0. CHECK LINE 127 IN preprocessing")
                                elif len(line) == 8:

                                    line.append("")
                                    line[8] = line[7]
                                    line[7] = line[6]

                                    temp = re.split('(-)', line[4])
                                    temp2 = re.split('(-)', line[5])

                                    while("" in temp):
                                        temp.remove("")
                                    while("" in temp2):
                                        temp2.remove("")
                                    
                                    if len(temp) > len(temp2):
                                        if len(temp) == 3:
                                            line[5] = temp[1] + temp[2]
                                            line[4] = temp[0]
                                        elif len(temp) == 4:
                                            line[5] = temp[2] + temp[3]
                                            line[4] = temp[0] + temp[1]
                                        else:
                                            print("There was an error auto correcting data field in line", line[0], "in protein", dir +"'s decoy", f)
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
                                            print("There was an error auto correcting data field in line", line[0], "in protein", dir +"'s decoy", f)
                                            print("This is the line:", line)
                                            raise Exception("WRONG LINE FORMAT-2. CHECK LINE 145 IN preprocessing")
                                    else:
                                        print("There was an error auto correcting data field in line", line[0], "in protein", dir +"'s decoy", f)
                                        print("This is the line:", line)
                                        raise Exception("WRONG LINE FORMAT-3. CHECK LINE 149 IN preprocessing")


                                #Here we finished fixing those weird lines. Now continue with data parsing
                                if len(line) == 9:
                                    #Just for debugging strange cases of abnormal atom counts
                                    if (line[7] != "1.00"):
                                        print("Protein", dir + "'s decoy", f + ", has an atom count of", line[7], "in line", line[0])

                                    try:
                                        #Every atom is a dictionary with it's own data
                                        atom = {
                                            "chain_type": line[1],
                                            "residue": line[2],
                                            "residue_number": int(line[3]),
                                            "coordinates": [float(line[4]), float(line[5]), float(line[6])],
                                            "atom_count": float(line[7]),
                                            "chain_type_single": line[8]
                                        }
                                        #We add the current atom to our atoms list
                                        atoms.append(atom)
                                    except:
                                        print("There was an error with the current line's data", line[0], "in protein", dir +", decoy", f)
                                        raise Exception("LINE DATA ERROR")
                                else:
                                    if line != []:
                                        print("Wrong line format", line[0], "in protein", dir +"'s decoy", f)
                                        print("This is the line:", line)
                                
                            #Here we went through all atoms. Now we add those atoms to the current decoy   
                            #Remember that f is the name of our file and the name of our decoy too
                            protein.addAtom(f, atoms)
                            
                    else:
                        with open(image_dir + "/" + dir + "/" + f) as file:

                            #For now, we only use the PDB to get data on the protein resolution.
                            resolutionFound = False
                            for line in file:
                                if "RESOLUTION" in line:
                                    resolutionFound = True

                                    #Here we remove everything we don't need from the current line
                                    line = line.split(" ")
                                    while ("" in line):
                                        line.remove("")
                                    if "\n" in line:
                                        line.remove("\n")
                                    if "\n" in line[-1]:
                                        line[-1] = line[-1].replace("\n", "")
                                    
                                    #Now we set the Resolution of our protein
                                    protein.setResolution(float(line[-1]))
                            
                            #Not having a resolution will jsut print debugging info to the console
                            if resolutionFound == False:
                                print("Didn't find Resolution for protein:", dir)    

                #We've gone through all data for the current protein. Now we add the protein instance to our protein list          
                proteins.append(protein)

    #Since we've finished reading all proteins, now we store them
    with open('all_protein_decoy_data.pickle', 'wb') as f:
        pickle.dump(proteins, f, -1)



if __name__ == "__main__":
    #Here we call our awesome parser :)
    readAndStoreData("CASP11Stage1_SCWRL")

    #This is to test data loading from the pickle file
    proteins = pickle.load(open("all_protein_decoy_data.pickle", "rb", -1))
    print(proteins[0].getDecoys()[0])
