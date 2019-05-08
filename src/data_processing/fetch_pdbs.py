'''
fetch_pdbs.py

Script fetches pdbs from Protein Data Bank as defined in class.csv files in
data_folder path. class.csv files includes pdb identifier and chain identifier
pairs. PDBs are save into folders corresponding to class and PDB identifier.

Script parallelizes fetches over multiple cores using MPI.

To run: $python3 fetch_pdbs.py or $mpirun -n N python3 fetch_pdbs.py

'''
import os, wget
import numpy as np
#from mpi4py import MPI
from random import randint

# Data folder path
data_folder = '../../data/KrasHras/'

###############################################################################
# TODO: get pdb updated list
# TODO: update pdb list
seed = 1234


def fetch_PDB(path, pdb_id):
    '''
    Method fetches pdb file from Protein Data Bank repo and stores file in
    designated path.
    '''
    # Download PDB file from database.
    url = 'https://files.rcsb.org/download/' # URL used to fetch PDB files
    file_path = path + pdb_id.lower() + '.pdb'
    if not os.path.exists(file_path):
        file_path = wget.download(url + pdb_id + '.pdb', out=path, bar=None)


def get_random_pbd_id(pdb_list = './source.idx'):
    with open(pdb_list) as f:
        content = f.readlines()

    rnd_num = randint(0, len(content))
    id = content[rnd_num][:4]
    return id

# TODO get PDB data sets


if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # MPI init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cores = comm.Get_size()

    # MPI task distribution
    if rank == 0:
        tasks = []

        # Iterate over PDB class .csv files
        for class_fn in sorted(os.listdir(data_folder)):
            if not class_fn.endswith('.csv'): continue

            # Make folder for PDB class
            class_ = class_fn.split('.')[0]
            if not os.path.exists(data_folder+class_): os.mkdir(data_folder+class_)

            # Iterate over PDB id and chain id pairs
            with open(data_folder+class_fn, 'r')as f:
                lines = f.readlines()
                for l in lines:

                    # Parse PDB IDs and chain IDs
                    l = l[:-1].split(',')
                    pdb_id = l[0].lower()
                    chain_id = l[1]
                    tasks.append([data_folder+class_, pdb_id, chain_id])

        # Shuffle for Random Distribution
        np.random.seed(seed)
        np.random.shuffle(tasks)

    else: tasks = None

    # Broadcast tasks to all nodes and select tasks according to rank
    tasks = comm.bcast(tasks, root=0)
    tasks = np.array_split(tasks, cores)[rank]

    # Fetch PDBs
    for t in tasks:

        # Task IDs
        folder_ = t[0]
        pdb_id = t[1]
        chain_id = t[2]

        # Fetch PDB file and rename with task IDs
        fetch_PDB(folder_, pdb_id)
        if not os.path.exists(folder_+'/'+ pdb_id +'_'+chain_id): os.mkdir(folder_+'/'+ pdb_id +'_'+chain_id)
        os.rename(folder_+'/'+pdb_id+'.pdb', folder_+'/'+pdb_id+'_'+chain_id+'/'+pdb_id+'_'+chain_id+'.pdb')

        print('Fetched: ' + pdb_id)