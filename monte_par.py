import subprocess
import json
import os
import numpy as np
import copy
from casm.project import Project
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# dividing data into chunks
def chunks(data,size):
    chunks = [[] for _ in range(size)]
    for i, chunk in enumerate(data):
        chunks[i % size].append(chunk)
    return chunks

# Grid space to run Monte Carlo calculations (min, max, delta)
T_grid = (200,841,20)
xi_grid = (-1.5,2.501,0.05)
if rank == 0:
    T_grid_params = np.arange(*T_grid)
    xi_grid_params = np.arange(*xi_grid)
    print(len(T_grid_params))
    print(int(np.ceil(len(T_grid_params)/size)))
	# separate grids for multiple cores
    T_grid_sep = list(chunks(T_grid_params,size)) 
    xi_grid_sep = list(chunks(xi_grid_params,size)) 
    indexes_T = np.arange(0,len(T_grid_params),1)
    indexes_xi = np.arange(0,len(xi_grid_params),1)
    indexes_T_sep = list(chunks(indexes_T,size)) 
    indexes_xi_sep = list(chunks(indexes_xi,size)) 
else:
	T_grid_sep, xi_grid_sep, indexes_T_sep, indexes_xi_sep = None, None, None, None
T_grid_sep = comm.scatter(T_grid_sep,root=0)
xi_grid_sep = comm.scatter(xi_grid_sep,root=0)
indexes_T_sep = comm.scatter(indexes_T_sep,root=0)
indexes_xi_sep = comm.scatter(indexes_xi_sep,root=0)
print(T_grid_sep)


d = {
  "mode" : "incremental",
  "dependent_runs" : True,
  "motif" : {
    "configname" : "restricted_auto",
  },
  "initial_conditions" : {
    "param_chem_pot" : {
      "a" : -2.00
    },
    "temperature" : 1200.0,
    "tolerance" : 0.001
  },
  "final_conditions" : {
    "param_chem_pot" : {
      "a" : 2.00
    },
    "temperature" : 10.0,
    "tolerance" : 0.001
  },
  "incremental_conditions" : {
    "param_chem_pot" : {
      "a" : 0.01
    },
    "temperature" : -10.0,
    "tolerance" : 0.001
  }
}

proj = None

def monte_cmd(input):
    subprocess.call(["casm monte -s input.json --verbosity quiet"],shell=True)
    pass
def submit_T_up(T_grid, xi_grid_sep):
  index = indexes_xi_sep[0]
  for xi in xi_grid_sep:
    path = "T_up." + str(index)
    
    tinput = copy.deepcopy(input)
    tinput['driver'] = d
    tinput['driver']['initial_conditions']['temperature'] = T_grid[0]
    tinput['driver']['final_conditions']['temperature'] = T_grid[1]
    tinput['driver']['incremental_conditions']['temperature'] = T_grid[2]
    
    tinput['driver']['initial_conditions']['param_chem_pot']['a'] = xi
    tinput['driver']['final_conditions']['param_chem_pot']['a'] = xi
    tinput['driver']['incremental_conditions']['param_chem_pot']['a'] = 0.0
    
    cwd = os.getcwd()
    try:
        os.mkdir(path)
    except OSError as error:
            print(error)
            pass
    os.chdir(path)
    
    with open('input.json','w') as f:
      json.dump(tinput, f)
    
    monte_cmd(tinput)
    
    os.chdir(cwd)
    index += 1

def submit_T_down(T_grid, xi_grid_sep):
  index = indexes_xi_sep[0]
  for xi in xi_grid_sep:
    path = "T_down." + str(index)
    
    tinput = copy.deepcopy(input)
    tinput['driver'] = d
    tinput['driver']['initial_conditions']['temperature'] = T_grid[1]
    tinput['driver']['final_conditions']['temperature'] = T_grid[0]
    tinput['driver']['incremental_conditions']['temperature'] = -T_grid[2]
    
    tinput['driver']['initial_conditions']['param_chem_pot']['a'] = xi
    tinput['driver']['final_conditions']['param_chem_pot']['a'] = xi
    tinput['driver']['incremental_conditions']['param_chem_pot']['a'] = 0.0
    
    cwd = os.getcwd()
    try:
        os.mkdir(path)
    except OSError as error:
            print(error)
            pass
    os.chdir(path)
    
    with open('input.json','w') as f:
      json.dump(tinput, f)
    
    monte_cmd(tinput)
    
    os.chdir(cwd)
    index += 1

def submit_xi_up(T_grid_sep, xi_grid):
  index = indexes_T_sep[0]
  for T in T_grid_sep:
    path = "xi_up." + str(index)
    
    tinput = copy.deepcopy(input)
    tinput['driver'] = d
    tinput['driver']['initial_conditions']['temperature'] = float(T)
    tinput['driver']['final_conditions']['temperature'] = float(T)
    tinput['driver']['incremental_conditions']['temperature'] = 0.0
    
    tinput['driver']['initial_conditions']['param_chem_pot']['a'] = xi_grid[0]
    tinput['driver']['final_conditions']['param_chem_pot']['a'] = xi_grid[1]
    tinput['driver']['incremental_conditions']['param_chem_pot']['a'] = xi_grid[2]
    
    cwd = os.getcwd()
    try:
        os.mkdir(path)
    except OSError as error:
            print(error)
            pass
    os.chdir(path)
    
    with open('input.json','w') as f:
      json.dump(tinput, f)
    
    monte_cmd(tinput)
    
    os.chdir(cwd)
    index += 1

def submit_xi_down(T_grid_sep, xi_grid):
  index = indexes_T_sep[0]
  for T in T_grid_sep:
    path = "xi_down." + str(index)
    
    tinput = copy.deepcopy(input)
    tinput['driver'] = d
    tinput['driver']['initial_conditions']['temperature'] = float(T)
    tinput['driver']['final_conditions']['temperature'] = float(T)
    tinput['driver']['incremental_conditions']['temperature'] = 0.0
    
    tinput['driver']['initial_conditions']['param_chem_pot']['a'] = xi_grid[1]
    tinput['driver']['final_conditions']['param_chem_pot']['a'] = xi_grid[0]
    tinput['driver']['incremental_conditions']['param_chem_pot']['a'] = -xi_grid[2]
    
    cwd = os.getcwd()
    try:
        os.mkdir(path)
    except OSError as error:
            print(error)
            pass
    os.chdir(path)
    print("rank:", rank, 'WORKING ini:', path, 'index', index) 
    with open('input.json','w') as f:
      json.dump(tinput, f)
    
    monte_cmd(tinput)
    
    os.chdir(cwd)
    index += 1



####  

with open('metropolis_grand_canonical.json','r') as f:
  input = json.load(f)

submit_xi_down(T_grid_sep, xi_grid)
submit_xi_up(T_grid_sep, xi_grid)
submit_T_up(T_grid, xi_grid_sep)
submit_T_down(T_grid, xi_grid_sep)
