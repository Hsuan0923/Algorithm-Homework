# -*- coding: utf-8 -*-
"""
Created on Wed May 12 09:39:37 2021

@author: User
"""
import random as rd
import copy
import matplotlib.pyplot as plt
import time
import sys
from itertools import combinations
from scipy.special import comb
import numpy as np
def SetCostMatrix(num):
    cmatrix = {}
    for i in range(1, num + 1):
        for j in range(i, num + 1):
            if i == j:
                cmatrix[(i, j)] = 0
            else:
                cmatrix[(i, j)] = np.random.randint(1, 30)
                cmatrix[(j, i)] = cmatrix[(i, j)]
    return cmatrix

def GetCostVal(row, col, source):
    if col == 0:
        col = source
        return num_cities[(row, col)]
    return num_cities[(row, col)]
def DP(a,aveTSP):
    D={}
    P={}
    #t = time.time()
    for i in range(2,node+1):
        D[(i,())] = b[i-1][0]
    #print(D)
    summ=sys.maxsize
    for k in range(1,node-1):
        B=list(combinations(a,k))
        #print("COMBINATION: ",B)
        for p in range(int(comb(node-1,k))): #子集合A的組數
            A=B[p]
            C=list(B[p])
            #print("A: ",A)
            for ii in range(len(a)-len(A)):
                i=set(a)-set(A)
                l=list(i)
                l.sort()
                i=l[ii]
                #print("i: ",i)
                total=sys.maxsize
                rou=[]
                for alen in range(len(A)):
                    j=A[alen]
                    #print("j: ",j)
                    if len(A)==1:
                        if total>(b[i-1][j-1]+D[(j,())]):
                            rou.append(j)  
                        total=min(total,b[i-1][j-1]+D[(j,())])
                    else:
                        reA=list(A)
                        reA.remove(j)
                        reA=tuple(reA)
                        #print(reA)
                        #ress = functools.reduce(lambda sub, ele: sub * 10 + ele, reA)
                        if total>(b[i-1][j-1]+D[(j,reA)]):
                            rou.append(j) 
                        total=min(total,b[i-1][j-1]+D[(j,reA)])
                        reA=list(reA)
                        reA.append(j)
                        reA.sort()
                #res = functools.reduce(lambda sub, ele: sub * 10 + ele, A)
                D[(i,A)]=total
                #print(rou)
                P[(i,A)]=rou[-1]
                #print("min: ",D[(i,res)])
    path=[1]
    rout=[]
    for k in range(2,node+1):
        rea=list(a)
        rea.remove(k)
        rea=tuple(rea)
        #print(reA)
        #resss = functools.reduce(lambda sub, ele: sub * 10 + ele, rea)
        if summ>(b[0][k-1]+D[(k,rea)]):
            rout.append(k)
        summ=min(summ,b[0][k-1]+D[(k,rea)])
        rea=list(rea)
        rea.append(k)
        rea.sort()
    #last = functools.reduce(lambda sub, ele: sub * 10 + ele, a)
    a=tuple(a)
    D[(1,a)]=summ
    P[(1,a)]=rout[-1]
    aveTSP+=D[(1,a)]
    print("Best(min) Weight: ",D[(1,a)])
    #runtime = round(time.time() - t, 3)
    #avetime+=runtime
    #print("run time: ",runtime)
    a=list(a)
    c=a[:]
    c=tuple(c)
    first=P[(1,c)]
    path.append(first)
    c=list(c)
    for m in range(node-2):
        c.remove(first)
        c=tuple(c)
        #rec = functools.reduce(lambda sub, ele: sub * 10 + ele, c)
        first=P[(first,c)]
        path.append(first)
        c=list(c)
    path.append(1)
    return path,aveTSP,summ
class Location:
    def __init__(self, name):
        self.loc = name
        #self.loc = (x, y)

    def distance_between(self, location2):
        assert isinstance(location2, Location)
        return b[self.loc-1][location2.loc-1]
class Route:
    def __init__(self, path):
        # path is a list of Location obj
        self.path = path
        self.length = self._set_length()

    def _set_length(self):
        total_length = 0
        path_copy = self.path[:]
        from_here = path_copy.pop(0)
        init_node = copy.deepcopy(from_here)
        while path_copy:
            to_there = path_copy.pop(0)
            total_length += to_there.distance_between(from_here)
            from_here = copy.deepcopy(to_there)
        total_length += from_here.distance_between(init_node)
        return total_length
class GeneticAlgo:
    def __init__(self, locs, level=10, populations=100, variant=3, mutate_percent=0.01, elite_save_percent=0.1):
        self.locs = locs
        self.level = level
        self.variant = variant
        self.populations = populations
        self.mutates = int(populations * mutate_percent)
        self.elite = int(populations * elite_save_percent)
    def _find_path(self):
        # locs is a list containing all the Location obj
        locs_copy = self.locs[:]
        path = []
        while locs_copy:
            to_there = locs_copy.pop(locs_copy.index(rd.choice(locs_copy)))
            path.append(to_there)
        return path

    def _init_routes(self):
        routes = []
        for _ in range(self.populations):
            path = self._find_path()
            routes.append(Route(path))
        return routes
    def _get_next_route(self, routes):
        routes.sort(key=lambda x: x.length, reverse=False)
        elites = routes[:self.elite][:]
        crossovers = self._crossover(elites)
        return crossovers[:] + elites

    def _crossover(self, elites):
        # Route is a class type
        normal_breeds = []
        mutate_ones = []
        for _ in range(self.populations - self.mutates):
            father, mother = rd.choices(elites[:4], k=2)
            index_start = rd.randrange(0, len(father.path) - self.variant - 1)
            # list of Location obj
            father_gene = father.path[index_start: index_start + self.variant]
            father_gene_names = [loc.loc for loc in father_gene]
            mother_gene = [gene for gene in mother.path if gene.loc not in father_gene_names]
            mother_gene_cut = rd.randrange(1, len(mother_gene))
            # create new route path
            next_route_path = mother_gene[:mother_gene_cut] + father_gene + mother_gene[mother_gene_cut:]
            next_route = Route(next_route_path)
            # add Route obj to normal_breeds
            normal_breeds.append(next_route)

            # for mutate purpose
            copy_father = copy.deepcopy(father)
            idx = range(len(copy_father.path))
            gene1, gene2 = rd.sample(idx, 2)
            copy_father.path[gene1], copy_father.path[gene2] = copy_father.path[gene2], copy_father.path[gene1]
            mutate_ones.append(copy_father)
        mutate_breeds = rd.choices(mutate_ones, k=self.mutates)
        return normal_breeds + mutate_breeds
    def evolution(self):
        routes = self._init_routes()
        for _ in range(self.level):
            routes = self._get_next_route(routes)
        routes.sort(key=lambda x: x.length)
        return routes[0].path, routes[0].length
def create_locations(a):
        locations = []
        #xs = [8, 50, 18, 35, 90, 40, 84, 74, 34, 40, 60, 74]
        #ys = [3, 62, 0, 25, 89, 71, 7, 29, 45, 65, 69, 47]
        cities = a
        for name in cities:
            locations.append(Location(name))
        return locations,cities
    
if __name__ == '__main__':
    optime_dp=[]
    optime_ga=[]
    opterror=[]
    a=[2,3]
    for node in range(4,21):
        avetime_dp=0
        avetime_ga=0
        aveTSP=0
        aveTSP_ga=0
        error=0
        total_num = node
        a.append(node)
        for re in range(1,6):
            num_cities = SetCostMatrix(total_num)
            b=[]
            for li in range(node):
                b.append([])
                for lis in range(node):
                    b[li].append(GetCostVal(li+1,lis+1,1))
            print(np.array(b))
            dpt = time.time()
            path,aveTSP,bestw=DP(a,aveTSP)
            dpruntime = round(time.time() - dpt, 3)
            avetime_dp+=dpruntime
            print("DP runtime: ",dpruntime)
            print("DP: ",path)

            
            gat = time.time()
            aa=a[:]
            aa.insert(0,1)
            my_locs,cities = create_locations(aa)
            my_algo = GeneticAlgo(my_locs, level=600, populations=150, variant=2, 
                                  mutate_percent=0.02, elite_save_percent=0.15)
            best_route, best_route_length = my_algo.evolution()
            best_route.append(best_route[0])
            #print([loc.loc for loc in best_route], best_route_length)
            garuntime = round(time.time() - gat, 3)
            avetime_ga+=garuntime
            aveTSP_ga+=best_route_length
            print("GA distance: ",best_route_length)
            print("GA runtime: ",garuntime)
            print("GA : ",[loc.loc for loc in best_route])
            err=(best_route_length-bestw)/bestw
            error+=err
            print("error: ",err)
            
        print("node: ", node)
        print("DP ave time: ",avetime_dp/5)
        optime_dp.append(avetime_dp/5)
        print("DP ave weight: ",aveTSP/5)
        print("GA ave time: ",avetime_ga/5)
        optime_ga.append(avetime_ga/5)
        print("GA ave weight: ",aveTSP_ga/5)
        print("ave error: ",error/5)
        opterror.append(error/5)
        print("")
    plt.figure(figsize=(6, 3))
    
    plt.subplot(1, 2, 1)
    a.remove(2)
    a.remove(3)
    plt.plot(a, optime_dp)
    plt.plot(a, optime_ga)
    plt.xlabel("node")
    plt.ylabel("runtime (seconds)")
    plt.legend(labels=["dp runtime", "ga runtime"])
    
    plt.subplot(1, 2, 2)
    plt.plot(a, opterror)
    plt.xlabel("node")
    plt.ylabel("error")
    #plt.legend(labels=["dp runtime", "ga runtime"])
    
    plt.tight_layout()

    
