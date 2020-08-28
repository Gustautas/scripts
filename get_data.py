import json
import pickle
import subprocess
from shutil import copy, copytree
import os
import pandas as pd
import fileinput
import pymatgen as pmg
import numpy as np


class Analazer:
    """A class for analyzing geometry of a structure"""
    def __init__(self,vasp_structure_path):
        """Read the structure, get symmetrized structure"""
        self.strc = pmg.Structure.from_file(vasp_structure_path)
        self.space_group = pmg.symmetry.analyzer.SpacegroupAnalyzer(self.strc).get_space_group_number()
        self.conv_strc = pmg.symmetry.analyzer.SpacegroupAnalyzer(self.strc).get_refined_structure() # conventinal symmetrized
        self.symm_strc = pmg.symmetry.analyzer.SpacegroupAnalyzer(self.strc).get_symmetrized_structure() # OG symmetrized
        ## Separate Na1 (corner Na atoms) and Na2 atoms and others
        self.Na1_s, self.Na2_s, self.Ti_s, self.Mn_s, self.P_s, self.O_s = ([] for i in range(6))
        for site in self.strc.sites:
            #print(site)
            #print(str(site.specie))
            if str(site.specie) == 'Na':
                #print('Ture')
                #print(' P ' in str(self.strc.get_neighbors(site,3.2)))
                if ' P ' in str(self.strc.get_neighbors(site,3.2)):
                    self.Na2_s.append(site)
                else:
                    self.Na1_s.append(site)
            elif str(site.specie) == 'Ti':
                self.Ti_s.append(site)
            elif str(site.specie) == 'Mn':
                self.Mn_s.append(site)
            elif str(site.specie) == 'P':
                self.P_s.append(site)
            elif str(site.specie) == 'O':
                self.O_s.append(site)
        #print('# of Na1:', len(self.Na1_s),'\n# of Na2:', len(self.Na2_s))
        self.separated_species = {'Na1':self.Na1_s, 'Na2':self.Na2_s, 'Ti':self.Ti_s, 'Mn':self.Mn_s, 'P':self.P_s, 'O':self.O_s}


    def get_spec_neighbours(self,site,species,r,coord_n,rmin=0.0):
        """Gets neighbours distances(Only specified atom type) in radius r"""
        neigbours = self.strc.get_neighbors(site,r)
        distances = []
        for neigbour in neigbours:
            if str(neigbour.specie) == species:
                d = neigbour.distance(site)
                if d > rmin:
                    distances.append(d)
        # leave only # of closest species
        distances = np.sort(distances)
        distances = distances[:coord_n]
        return distances

    def get_neighbours_dict(self,specie1,specie2,r,coord_n,rmin=0.0):
        """Gets neighbours dict (Only specified atom type in radius r)
        with coords, distances, average dist, deviations"""
        n_dict = {}
        n_dict['Coordination'] = specie1 + ' - ' + specie2
        index = 0
        for site in self.separated_species[specie1]:
            n_dict[index] = {}
            n_dict[index]['coords'] = site.frac_coords
            n_dict[index]['distances'] = self.get_spec_neighbours(site,specie2,r,coord_n)
            n_dict[index]['local_mean_d'] = np.mean(n_dict[index]['distances'])
            n_dict[index]['local_deviation'] = np.std(n_dict[index]['distances'])
            index += 1
        # mean_local deviation
        # global_mean
        local_deviations = []
        local_means = []
        number_of_neigbours =[]
        for d in n_dict:
            if type(d) is int:
                local_deviations.append(n_dict[d]['local_deviation'])
                local_means.append(n_dict[d]['local_mean_d'])
                number_of_neigbours.append(len(n_dict[d]['distances']))
        n_dict['mean_local_deviation'] = np.mean(local_deviations)
        n_dict['global_mean'] = np.mean(local_means)
        # global_mean_deviation
        n_dict['global_mean_deviation'] = np.std(local_means)
        # avg number of neigbours
        n_dict['avg_#_of_neighbours'] = np.mean(number_of_neigbours)

        return n_dict



parant_path = os.path.abspath(os.getcwd())
configs = pd.read_csv('configs.csv', delim_whitespace=True)
names = configs['configname']



Dict = {}
for name in names:
    calc_dir = "../training_data/"+name+"/calctype.default/run.final"
    Dict[name]={}
    Dict[name]['x']= float(configs.loc[names == name, 'comp(a)']*1.5)
    Dict[name]['dU']= float(configs.loc[names == name, 'formation_energy']/2)
    Dict[name]['#_F.U.']= int(configs.loc[names == name, 'scel_size']*2)
   
    S = Analazer(calc_dir+"/CONTCAR")
    Dict[name]['pmg_struct']= S.strc
    Dict[name]['bonds']={}
    Dict[name]['bonds']['Na1-O']=S.get_neighbours_dict('Na1','O',r=3.0,coord_n=6,rmin=0.0)
    Dict[name]['bonds']['Na2-O']=S.get_neighbours_dict('Na2','O',r=5.0,coord_n=10,rmin=0.0)
    Dict[name]['bonds']['P-O']=S.get_neighbours_dict('P','O',r=3.0,coord_n=4,rmin=0.0)
    Dict[name]['bonds']['Ti-O']=S.get_neighbours_dict('Ti','O',r=4.0,coord_n=6,rmin=0.0)
    Dict[name]['bonds']['Mn-O']=S.get_neighbours_dict('Mn','O',r=4.0,coord_n=6,rmin=0.0)
    Dict[name]['bonds']['Ti-Ti first_near_site']=S.get_neighbours_dict(
                             'Ti','Ti',r=5.55,coord_n=1,rmin=0.0)
    Dict[name]['bonds']['Ti-Mn first_near_site']=S.get_neighbours_dict(
                             'Ti','Mn',r=5.55,coord_n=1,rmin=0.0)
    Dict[name]['bonds']['Mn-Mn first_near_site']=S.get_neighbours_dict(
                             'Mn','Mn',r=5.55,coord_n=1,rmin=0.0)

with open('data.pkl', 'wb') as file:
    pickle.dump(Dict, file)

