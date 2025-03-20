# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 19:23:34 2023

@author: jlegalla
"""
import ERMESS_classes as ECl
import ERMESS_operators as Eop
import numpy as np
import copy
import time
from itertools import chain
from numba import jit
import pickle
    
        
def solution_changing_scale(inputs):
    (D_solutions,D_time_resolution,time_resolution,D_movable_load,n_bits,n_store)=(inputs[i] for i in range(6))
    solutions=[]
    for S in D_solutions :
        solution=ECl.Individual(production_set=S.production_set,storage_sum=S.storage_sum,storage_TS=np.zeros((len(S.storage_TS),n_bits)),contract=S.contract,Y_DSM=np.zeros(n_bits),D_DSM=np.zeros((int(n_bits/time_resolution/24),int(time_resolution*24))),fitness=np.NAN)
        for i in range(len(S.storage_TS[0,:])):
            reparti=S.storage_TS[:,i]
            for j in range(n_store):
                solution.storage_TS[j,int(i*time_resolution*24):int((i+1)*time_resolution*24)]=np.random.uniform(0,1,(int(time_resolution*24)))
                solution.storage_TS[j,int(i*time_resolution*24):int((i+1)*time_resolution*24)]=solution.storage_TS[j,int(i*time_resolution*24):int((i+1)*time_resolution*24)]*reparti[j]/np.mean(solution.storage_TS[j,int(i*time_resolution*24):int((i+1)*time_resolution*24)])
            solution.Y_DSM[int(i*time_resolution*24):int((i+1)*time_resolution*24)]=np.random.uniform(0,1,(int(time_resolution*24)))
            solution.Y_DSM[int(i*time_resolution*24):int((i+1)*time_resolution*24)]=solution.Y_DSM[int(i*time_resolution*24):int((i+1)*time_resolution*24)]/np.mean(solution.Y_DSM[int(i*time_resolution*24):int((i+1)*time_resolution*24)])*S.Y_DSM[i]
            solution.D_DSM[i]=np.random.uniform(0,1,(int(time_resolution*24)))
            solution.D_DSM[i]=solution.D_DSM[i]/sum(solution.D_DSM[i])*sum(D_movable_load[int(i*time_resolution*24):int((i+1)*time_resolution*24)])
        solutions.append(solution)
    return (solutions)

def solution_extending_scale(inputs):
    (H_world_population3,D_movable_load,Y_movable_load,time_resolution,H_duration_years,duration_years,n_bits,n_store)=(inputs[i] for i in range(8))
    extending_pop=[]
    duplicate_index = duration_years/H_duration_years
    for ind in H_world_population3:
        extended_ind=ECl.Individual(production_set=ind.production_set,storage_sum=ind.storage_sum,storage_TS=ind.storage_TS,contract=ind.contract,Y_DSM=ind.Y_DSM,D_DSM=ind.D_DSM,fitness=np.NAN)
        #extended_ind.storage_sum=[extended_ind.storage_sum[i]*duplicate_index for i in range(len(extended_ind.storage_sum))]
        test=[]
        extended_ind.D_DSM=np.array([extended_ind.D_DSM[0] for i in range(int(duplicate_index))])
        for i in range(len(extended_ind.storage_TS)):
            test.append(extended_ind.storage_TS[i])
            for _ in range(int(duplicate_index)-1):
                test[i]=np.append(test[i],extended_ind.storage_TS[i])
        extended_ind.storage_TS=np.array(test)
        extended_ind.Y_DSM=np.random.rand(n_bits)
        extended_ind.Y_DSM=extended_ind.Y_DSM/sum(extended_ind.Y_DSM)*sum(Y_movable_load)
        
        for i in range(int(duplicate_index)):
            extended_ind.D_DSM[i]=extended_ind.D_DSM[0]/(0.01+sum(extended_ind.D_DSM[0]))*sum(D_movable_load[int(i*time_resolution*24):int((i+1)*time_resolution*24)])
        extended_ind.fitness=np.NAN
        extended_ind.storage_sum=-np.sum(np.where(extended_ind.storage_TS<0,extended_ind.storage_TS,0),axis=1)
        
        extending_pop.append(extended_ind)
    return(extending_pop)

        

# tournament selection
def selection(pop, scores, k=3):
 # first random selection
    selection_ix = np.random.randint(len(pop))
    for ix in np.random.randint(0, len(pop), k-1):
 # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

# tournament selection
def selection2(pop, k=3):
 # first random selection
    selection_ix = np.random.randint(len(pop))
    for ix in np.random.randint(0, len(pop), k-1):
 # check if better (e.g. perform a tournament)
        if pop[ix].fitness < pop[selection_ix].fitness:
            selection_ix = ix
    return pop[selection_ix]

def crossover_avec_pertes_contraintes_3(p1, p2, r_cross,n_bits, n_store,storage_characteristics):
 
    # children are copies of parents by default

    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

    c1.fitness=np.NAN
    c2.fitness=np.NAN

 # check for recombination
    if np.random.rand() < r_cross:
    # select random weights

        weights = np.random.random(2*len(c1.production_set))
 # perform crossover
        c1.production_set=(weights[0:len(c1.production_set)]*p1.production_set+(1-weights[0:len(c1.production_set)])*p2.production_set).round().astype(int)
        c2.production_set=(weights[(len(c1.production_set)):(2*len(c1.production_set))]*p2.production_set+(1-weights[(len(c1.production_set)):(2*len(c1.production_set))])*p1.production_set).round().astype(int)

        #choices = np.random.choice([0,1],(2,len(c1.storage_sum)+1))
 # Si méthode ancienne
        #c1.storage_sum=[c1.storage_sum[j] if choices[0,j]==0 else c2.storage_sum[j] for j in range(len(c1.storage_sum))]
        #c2.storage_sum=[c1.storage_sum[j] if choices[1,j]==0 else c2.storage_sum[j] for j in range(len(c1.storage_sum))]
        choices = np.random.choice([0,1],2)
        weights = np.random.random(2*n_store)
        
       # c1.storage_TS=np.array([p1.storage_TS[i]*weights[i]+p2.storage_TS[i]*(1-weights[i]) for i in range(n_store)])
        c1.storage_TS=np.transpose(np.multiply(weights[0:n_store],np.transpose(p1.storage_TS))+np.multiply((1-weights[0:n_store]),np.transpose(p2.storage_TS)))
       # c2.storage_TS=np.array([p1.storage_TS[i]*weights[n_store+i]+p2.storage_TS[i]*(1-weights[n_store+i]) for i in range(n_store)])
        c2.storage_TS=np.transpose(np.multiply(weights[(n_store+1):len(weights)],np.transpose(p1.storage_TS))+np.multiply((1-weights[(n_store+1):len(weights)]),np.transpose(p2.storage_TS)))
        
        
        c1.contract=c1.contract if choices[0]==0 else c2.contract
        c2.contract=c1.contract if choices[1]==0 else c2.contract
        
        weights = np.random.random(len(c1.D_DSM)+1)
        c1.Y_DSM = p1.Y_DSM*weights[len(c1.D_DSM)]+p2.Y_DSM*(1-weights[len(c1.D_DSM)])
        c2.Y_DSM = p2.Y_DSM*weights[len(c1.D_DSM)]+p1.Y_DSM*(1-weights[len(c1.D_DSM)])
        c1.D_DSM = (np.multiply(p1.D_DSM.T,weights[0:len(c1.D_DSM)])+np.multiply(p2.D_DSM.T,(1-weights[0:len(c1.D_DSM)]))).T
        c2.D_DSM = (np.multiply(p2.D_DSM.T,weights[0:len(c1.D_DSM)])+np.multiply(p1.D_DSM.T,(1-weights[0:len(c1.D_DSM)]))).T
        
        ##Si méthode génétique
        
 #       p1c2=np.random.choice(n_bits,np.random.randint(1,n_bits),replace=False)
 #       for i in range(n_store):
 #           c1[2][i][p1c2]=p2[2][i][p1c2]
 #           c1[1][i]=0 if ((all(val <= 0 for val in c1[2][i]))|((all(val >= 0 for val in c1[2][i])))) else -sum(c1[2][i][c1[2][i]<0])
 #           c2[2][i][p1c2]=p1[2][i][p1c2]
 #           c2[1][i]=0 if ((all(val <= 0 for val in c2[2][i]))|((all(val >= 0 for val in c2[2][i])))) else -sum(c2[2][i][c2[2][i]<0])

        
    #On introduit une mutation éventuelles des quantités
  ##  if ((np.random.rand()<r_cross/6)) :
  ##      c1.storage_TS=np.multiply(np.random.uniform(0.9,1.1,n_store),c1.storage_TS.T).T
  ##      c2.storage_TS=np.multiply(np.random.uniform(0.9,1.1,n_store),c2.storage_TS.T).T
   # for i in range(n_store) :
   #     if ((np.random.rand()<r_cross/6)) :
        
   #         c1.storage_TS[i]=c1.storage_TS[i]*np.random.uniform(0.9,1.1,1)
   #         c2.storage_TS[i]=c2.storage_TS[i]*np.random.uniform(0.9,1.1,1)     
          
    return [c1, c2]


def crossover_decalage(p1, p2, r_cross,n_bits, n_store,storage_characteristics):
 
    # children are copies of parents by default

    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

    c1.fitness=np.NAN
    c2.fitness=np.NAN

 # check for recombination
    if np.random.rand() < r_cross:
    # select random weights

        weights = np.random.random(1)
 # perform crossover
        c1.production_set=(weights[0]*p1.production_set+(1-weights[0])*p2.production_set).round().astype(int)
        c2.production_set=(weights[0]*p2.production_set+(1-weights[0])*p1.production_set).round().astype(int)

        breaks = np.sort(np.random.choice(range(1,len(p1.storage_TS[0])-1),3)  )            
       
        c1.storage_TS=np.concatenate((p1.storage_TS[:,np.arange(0,breaks[0])],p2.storage_TS[:,np.arange(breaks[0],breaks[1])],p1.storage_TS[:,np.arange(breaks[1],breaks[2])],p2.storage_TS[:,np.arange(breaks[2],len(p1.storage_TS[0]))]),axis=1)
        c2.storage_TS=np.concatenate((p2.storage_TS[:,np.arange(0,breaks[0])],p1.storage_TS[:,np.arange(breaks[0],breaks[1])],p2.storage_TS[:,np.arange(breaks[1],breaks[2])],p1.storage_TS[:,np.arange(breaks[2],len(p1.storage_TS[0]))]),axis=1)
        
        
        c1.contract=c1.contract if weights[0]>0.5 else c2.contract
        c2.contract=c1.contract if weights[0]<0.5 else c2.contract
        
        weights = np.random.random(len(c1.D_DSM)+1)
        c1.Y_DSM = p1.Y_DSM*weights[len(c1.D_DSM)]+p2.Y_DSM*(1-weights[len(c1.D_DSM)])
        c2.Y_DSM = p2.Y_DSM*weights[len(c1.D_DSM)]+p1.Y_DSM*(1-weights[len(c1.D_DSM)])
        c1.D_DSM = (np.multiply(p1.D_DSM.T,weights[0:len(c1.D_DSM)])+np.multiply(p2.D_DSM.T,(1-weights[0:len(c1.D_DSM)]))).T
        c2.D_DSM = (np.multiply(p2.D_DSM.T,weights[0:len(c1.D_DSM)])+np.multiply(p1.D_DSM.T,(1-weights[0:len(c1.D_DSM)]))).T
          
    return [c1, c2]

def crossover_reduit_nonJIT(p1, p2, r_cross,n_bits, n_store,storage_characteristics):
 
    # children are copies of parents by default

    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

    c1.fitness=np.nan
    c2.fitness=np.nan
    cross_rand = np.random.rand()

 # check for recombination
    if cross_rand < r_cross:
    # select random weights

        weights = np.random.random(1)
 # perform crossover
        c1.production_set=(weights[0]*p1.production_set+(1-weights[0])*p2.production_set).round().astype(int)
        c2.production_set=(weights[0]*p2.production_set+(1-weights[0])*p1.production_set).round().astype(int)

        choices = np.random.choice([0,1],2)
        weights = np.random.random(2*n_store)
        
        c1.storage_TS=(weights[0]*p1.storage_TS+(1-weights[0])*p2.storage_TS)
        c2.storage_TS=(weights[0]*p2.storage_TS+(1-weights[0])*p1.storage_TS)
        
        
        c1.contract=c1.contract if weights[0]>0.5 else c2.contract
        c2.contract=c1.contract if weights[0]<0.5 else c2.contract
        
        weights = np.random.random(len(c1.D_DSM)+1)
        c1.Y_DSM = p1.Y_DSM*weights[len(c1.D_DSM)]+p2.Y_DSM*(1-weights[len(c1.D_DSM)])
        c2.Y_DSM = p2.Y_DSM*weights[len(c1.D_DSM)]+p1.Y_DSM*(1-weights[len(c1.D_DSM)])
        c1.D_DSM = (np.multiply(p1.D_DSM.T,weights[0:len(c1.D_DSM)])+np.multiply(p2.D_DSM.T,(1-weights[0:len(c1.D_DSM)]))).T
        c2.D_DSM = (np.multiply(p2.D_DSM.T,weights[0:len(c1.D_DSM)])+np.multiply(p1.D_DSM.T,(1-weights[0:len(c1.D_DSM)]))).T
          
    return [c1, c2,int(cross_rand<r_cross)]

@jit(nopython=True)
def crossover_reduit(p1, p2, r_cross,n_bits, n_store,storage_characteristics):
 
    # children are copies of parents by default

    c1 = ECl.Individual(p1.production_set,p1.storage_sum,p1.storage_TS,p1.contract,p1.Y_DSM,p1.D_DSM,np.float64(np.nan),np.empty(n_bits)* np.float64(np.nan)) 
    c2 = ECl.Individual(p2.production_set,p2.storage_sum,p2.storage_TS,p2.contract,p2.Y_DSM,p2.D_DSM,np.float64(np.nan),np.empty(n_bits)* np.float64(np.nan)) 

    cross_rand = np.random.rand()

 # check for recombination
    if cross_rand < r_cross:
    # select random weights

        weights = np.random.random(5)
 # perform crossover
        c1.production_set=np.array([round(weights[4]*p1.production_set[i]+(1-weights[4])*p2.production_set[i]) for i in range(len(p1.production_set))],dtype=np.int64)
        c2.production_set=np.array([round(weights[4]*p2.production_set[i]+(1-weights[4])*p1.production_set[i]) for i in range(len(p1.production_set))],dtype=np.int64)
##        c1.production_set=np.int64((np.around(weights*p1.production_set+(1-weights)*p2.production_set)))
#        c1.production_set=np.int64((weights[0]*p1.production_set+(1-weights[0])*p2.production_set).round())
#        c2.production_set=(weights[0]*p2.production_set+(1-weights[0])*p1.production_set).round().astype(int)

        
        
        c1.storage_TS=(weights[0]*p1.storage_TS+(1-weights[0])*p2.storage_TS)
        c2.storage_TS=(weights[0]*p2.storage_TS+(1-weights[0])*p1.storage_TS)
        
        
        c1.contract=c1.contract if weights[1]<0.5 else c2.contract
        c2.contract=c1.contract if weights[2]<0.5 else c2.contract
        
        c1.Y_DSM = p1.Y_DSM*weights[3]+p2.Y_DSM*(1-weights[3])
        c2.Y_DSM = p2.Y_DSM*weights[3]+p1.Y_DSM*(1-weights[3])
        
        weights = np.random.random(len(c1.D_DSM)+1)
        
        c1.D_DSM = (np.multiply(p1.D_DSM.T,weights[0:len(c1.D_DSM)])+np.multiply(p2.D_DSM.T,(1-weights[0:len(c1.D_DSM)]))).T
        c2.D_DSM = (np.multiply(p2.D_DSM.T,weights[0:len(c1.D_DSM)])+np.multiply(p1.D_DSM.T,(1-weights[0:len(c1.D_DSM)]))).T
          
    return (c1, c2,int(cross_rand<r_cross))



def crossover_complet(p1, p2, r_cross,n_bits, n_store,storage_characteristics):
    # children are copies of parents by default

    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

    c1.fitness=np.NAN
    c2.fitness=np.NAN

 # check for recombination
    if np.random.rand() < r_cross:
    # select random weights

        weights = np.random.random(2*len(c1.production_set)+len(c1.storage_TS[0]))
 # perform crossover
        c1.production_set=(weights[0:len(c1.production_set)]*p1.production_set+(1-weights[0:len(c1.production_set)])*p2.production_set).round().astype(int)
        c2.production_set=(weights[(len(c1.production_set)):(2*len(c1.production_set))]*p2.production_set+(1-weights[(len(c1.production_set)):(2*len(c1.production_set))])*p1.production_set).round().astype(int)

        choices = np.random.choice([0,1],2)
        
        c1.storage_TS=np.multiply(weights[(2*len(c1.production_set)):len(weights)],p1.storage_TS)+np.multiply(1-weights[(2*len(c1.production_set)):len(weights)],p2.storage_TS)
        c2.storage_TS=np.multiply(1-weights[(2*len(c1.production_set)):len(weights)],p1.storage_TS)+np.multiply(weights[(2*len(c1.production_set)):len(weights)],p2.storage_TS)
        
        
        c1.contract=c1.contract if choices[0]==0 else c2.contract
        c2.contract=c1.contract if choices[1]==0 else c2.contract
        
        weights = np.random.random(len(c1.D_DSM)+1)
        c1.Y_DSM = p1.Y_DSM*weights[len(c1.D_DSM)]+p2.Y_DSM*(1-weights[len(c1.D_DSM)])
        c2.Y_DSM = p2.Y_DSM*weights[len(c1.D_DSM)]+p1.Y_DSM*(1-weights[len(c1.D_DSM)])
        c1.D_DSM = (np.multiply(p1.D_DSM.T,weights[0:len(c1.D_DSM)])+np.multiply(p2.D_DSM.T,(1-weights[0:len(c1.D_DSM)]))).T
        c2.D_DSM = (np.multiply(p2.D_DSM.T,weights[0:len(c1.D_DSM)])+np.multiply(p1.D_DSM.T,(1-weights[0:len(c1.D_DSM)]))).T
          
    return [c1, c2]

    
# mutation operator
def mutation_contraintes_2(c, Bounds_prod, Non_movable_load, self_sufficiency, prods_U,prod_C, n_store, n_contracts):
    
       t0=time.time()
       if ((np.random.rand()<0.015)) :
            c.contract=np.random.randint(0,n_contracts,1)[0]
    #Mutation de la production
       c.production_set=np.array([min(Bounds_prod[i],max(0,c.production_set[i]+np.random.randint(min(-1,-int(Bounds_prod[i]/40)),max(1,int(Bounds_prod[i]/40)),1)[0])) for i in range(len(Bounds_prod))])
 # Mutation de la série temporelle
       t1=time.time()
       mutations=[np.random.choice(len(c.storage_TS[0]), size=np.random.randint(0,int(len(c.storage_TS[0])/20)+2), replace=False) for i in range(n_store)]
 # flip the bit
       t2=time.time()
       for store_mutations in range(len(mutations)) :
           for m in mutations[store_mutations] :          
               alter = np.random.choice(np.where(c.storage_TS[store_mutations]>=0 if c.storage_TS[store_mutations][m]>=0 else c.storage_TS[store_mutations]<0)[0], size=1)[0]
               change = min(abs(c.storage_TS[store_mutations][m]),abs(c.storage_TS[store_mutations][alter]))*np.random.uniform(-1,1)
               c.storage_TS[store_mutations][m]=c.storage_TS[store_mutations][m]+change
               c.storage_TS[store_mutations][alter]=c.storage_TS[store_mutations][alter]-change
       t3=time.time()        
       production = np.sum(np.array([c.production_set[i]*prods_U[i,:] for i in range(len(c.production_set))])/1000,axis=0)+prod_C/1000
       t4=time.time()
       importation = Non_movable_load-production-np.sum(c.storage_TS,axis=0)
       t5=time.time()
       importation[importation<0]=0
       t6=time.time()
       obtained_self_sufficiency = 1-(sum(importation)/sum(Non_movable_load))

       
       if (obtained_self_sufficiency < self_sufficiency) & (sum([c.storage_sum[i]>0 for i in range(n_store)])>0) :
           store=np.random.choice(np.where(np.array(c.storage_sum)>0)[0],1)[0]
           mutations=np.random.choice(np.where(importation>max(importation)*0.5)[0], size=np.random.randint(0,min(15,len(np.where(importation>max(importation)*0.5)[0]))), replace=False)
           for m in mutations:   
               alter = np.random.choice(np.where(c.storage_TS[store]>=0 if c.storage_TS[store][m]>=0 else  c.storage_TS[store]<=0)[0], size=1)[0]
               change = min(abs(c.storage_TS[store][m]),abs(c.storage_TS[store][alter]))*np.random.uniform(0.6,1)
               c.storage_TS[store][m]=c.storage_TS[store][m]+change 
               c.storage_TS[store][alter]=c.storage_TS[store][alter]-change      
       t7=time.time()
       print(t1-t0,t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6)
       return(c)
   
def mutation_contraintes_3_original(c, Bounds_prod, Non_movable_load, self_sufficiency, prods_U,prod_C, n_store, n_contracts, storage_characteristics):
       
        #Changement du contrat
          if ((np.random.rand()<0.003)) :   
              c.contract=np.random.randint(0,n_contracts,1)[0]
          if ((np.random.rand()<0.2)) :
       #Mutation de la production
              new_set = np.array([min(Bounds_prod[i],max(0,c.production_set[i]+np.random.randint(min(-1,-int(Bounds_prod[i]/20)),max(1,int(Bounds_prod[i]/20)),1)[0])) for i in range(len(Bounds_prod))])
              added_prod = np.dot((new_set - c.production_set),prods_U)*np.random.normal(1,0.05,len(Non_movable_load))
              c.production_set=new_set

              repartition=np.random.rand(n_store)
              c.storage_TS = np.subtract(c.storage_TS,np.outer(added_prod,repartition).T/np.sum(repartition)/1000)
              #facteur=np.random.rand(1)+0.5
              
              #for i in range(n_store):
              #    c.storage_TS[i]=c.storage_TS[i]*facteur
    # Mutation aléatoire de la série temporelle
          if ((np.random.rand()<0.6)) :
              #mutations=[ np.random.choice(range(len(c.storage_TS[i])), size=np.random.randint(1,int(len(c.storage_TS[i])/200)+2), replace=False) for i in range(n_store)]
              #for store_mutations in range(len(mutations)) :
              #    c.storage_TS[store_mutations][mutations[store_mutations]]=c.storage_TS[store_mutations][mutations[store_mutations]]*np.random.uniform(0.6,1.4)
              direction = np.random.randint(2)
              sub_storage_TS = [np.where(c.storage_TS[i]<0)[0] if direction==0 else np.where(c.storage_TS[i]>0)[0] for i in range(n_store)]
              mutations=[[] if len(sub_storage_TS[i])<5 else np.random.choice(sub_storage_TS[i], size=np.random.randint(1,int(len(sub_storage_TS[i])/200)+2), replace=False) for i in range(n_store)]
              
              for store_mutations in range(len(mutations)) :
                  random_index=np.random.random(len(mutations[store_mutations]))
                  c.storage_TS[store_mutations][mutations[store_mutations]]=random_index/sum(random_index)*sum(c.storage_TS[store_mutations][mutations[store_mutations]])

        #Mutation du DSM 
          if ((np.random.rand()<0.9)) :
          #if ((np.random.rand()<0.01)) :
              mutations_Y_DSM=np.random.choice(range(len(c.Y_DSM)), size=np.random.randint(1,int(len(c.Y_DSM)/50)+2), replace=False)
                  # flip the bit         
              random_index=np.random.uniform(0.3,3,len(mutations_Y_DSM))
              c.Y_DSM[mutations_Y_DSM]=random_index*c.Y_DSM[mutations_Y_DSM]*sum(c.Y_DSM[mutations_Y_DSM])/(sum(random_index*c.Y_DSM[mutations_Y_DSM]) if sum(c.Y_DSM[mutations_Y_DSM])!=0 else 1)
              #c.Y_DSM[mutations_Y_DSM]=random_index*c.Y_DSM[mutations_Y_DSM]

           #   days_mutations_DSM = np.random.choice(len(c.D_DSM),int(len(c.D_DSM)/30))
           #   for day in days_mutations_DSM:
           #       mutations_D_DSM=np.random.choice(range(len(c.D_DSM[day,])), size=np.random.randint(1,int(len(c.D_DSM[day,])/20)+2), replace=False)
                      # flip the bit         
           #       random_index=np.random.random(len(mutations_D_DSM))
           #       c.D_DSM[day,mutations_D_DSM]=random_index/sum(random_index)*sum(c.D_DSM[day,mutations_D_DSM])
          if ((np.random.rand()<0.2)) :
              mutations_D_DSM=np.random.choice(range(len(c.D_DSM[0,])), size=np.random.randint(1,int(len(c.D_DSM[0,])/10)+2), replace=False)
              random_index=np.random.random(len(mutations_D_DSM))
              for day in range(len(c.D_DSM)):
                  if ((np.random.rand()<0.4)) :
                      c.D_DSM[day,mutations_D_DSM]=random_index/sum(random_index)*sum(c.D_DSM[day,mutations_D_DSM])
                    
          #On introduit un transfert éventuel entre stockages
          if ((np.random.rand()<0.2)) :
              if ((np.random.rand()<0.5)) :
             
                  store_in1=np.random.choice(n_store)
                  store_out1=np.random.choice(n_store)
                  if (store_in1 != store_out1) :
                     switch_indexes1=np.random.choice(len(c.storage_TS[0]) ,max(2,int(len(c.storage_TS[0])/200)),replace=False)

                     sums = c.storage_TS[store_in1][switch_indexes1]+ c.storage_TS[store_out1][switch_indexes1]
                     ratios = np.random.uniform(-0.2,1.2,len(switch_indexes1))
                     c.storage_TS[store_in1][switch_indexes1]=sums*ratios
                     c.storage_TS[store_out1][switch_indexes1]=sums*(1-ratios)
              else :
                          
                  store_in1=np.random.choice(n_store)
                  store_out1=np.random.choice(n_store)
                  if (store_in1 != store_out1) :
                     switch_indexes1=np.random.choice(len(c.storage_TS[0]) ,max(2,int(len(c.storage_TS[0])/200)),replace=False)

                     c.storage_TS[store_in1][switch_indexes1]=c.storage_TS[store_in1][switch_indexes1]+ c.storage_TS[store_out1][switch_indexes1]
                     c.storage_TS[store_out1][switch_indexes1]=np.repeat(0,len(switch_indexes1))

          production = np.sum(((np.dot(c.production_set,prods_U),prod_C)),axis=0)/1000
          importation = Non_movable_load+np.concatenate(c.D_DSM)+c.Y_DSM-production-np.sum(c.storage_TS,axis=0)
          obtained_self_sufficiency = 1-(np.sum(np.where(importation>0,importation,0))/(np.sum(Non_movable_load)+np.sum(c.Y_DSM)+np.sum(c.D_DSM)))

        #On créé un opérateur qui engendre ou annule des mouvements opposés d'un stockage
          if ((np.random.rand()<0.5) & any(np.array([any(c.storage_TS[i]>0)  & any(c.storage_TS[i]<0) for i in range(len(c.storage_sum))])) ) :
              store_opp = np.random.choice(np.where(np.array([any(c.storage_TS[i]>0)  & any(c.storage_TS[i]<0) for i in range(len(c.storage_sum))]))[0],1)[0]
              sub_storage_opposite = [np.random.choice(np.where(c.storage_TS[store_opp]<0)[0],np.random.randint(1,max(2,int(0.04*len(np.where(c.storage_TS[store_opp]<0)[0])))),replace=False),np.random.choice(np.where(c.storage_TS[store_opp]>0)[0],np.random.randint(1,max(2,int(0.04*len(np.where(c.storage_TS[store_opp]>0)[0])))),replace=False)] 
              inflation= np.random.uniform(0.7,1.3) if obtained_self_sufficiency<self_sufficiency else np.random.uniform(0.4,1.05)
              sens_limite=np.argmin([-sum(c.storage_TS[store_opp][sub_storage_opposite[0]]),sum(c.storage_TS[store_opp][sub_storage_opposite[1]])/storage_characteristics['Round-trip efficiency'][store_opp]])
              creation = abs(sum(c.storage_TS[store_opp][sub_storage_opposite[sens_limite]]))*(inflation-1)
              c.storage_TS[store_opp][sub_storage_opposite[sens_limite]]=c.storage_TS[store_opp][sub_storage_opposite[sens_limite]]*inflation
              if (sens_limite==0):
                c.storage_TS[store_opp][sub_storage_opposite[1]]=c.storage_TS[store_opp][sub_storage_opposite[1]]+c.storage_TS[store_opp][sub_storage_opposite[1]]/sum(c.storage_TS[store_opp][sub_storage_opposite[1]])*creation*storage_characteristics['Round-trip efficiency'][store_opp]
              else :
                c.storage_TS[store_opp][sub_storage_opposite[0]]=c.storage_TS[store_opp][sub_storage_opposite[0]]-c.storage_TS[store_opp][sub_storage_opposite[0]]/sum(c.storage_TS[store_opp][sub_storage_opposite[0]])*creation/storage_characteristics['Round-trip efficiency'][store_opp]
              #c.storage_sum[store_opp]=-sum(c.storage_TS[store_opp][c.storage_TS[store_opp]<0])
        
          if ((obtained_self_sufficiency < self_sufficiency)|(np.random.rand()<0.1)) & (sum([c.storage_sum[i]>0 for i in range(n_store)])>0) & sum(c.production_set)>0:
              if ((np.random.rand()<0.4)) :
                  production_set_subs=np.array([min(Bounds_prod[i],max(0,c.production_set[i]+np.random.randint(0,max(2,int(Bounds_prod[i]/10)),1)[0])) for i in range(len(Bounds_prod))])
                  facteur_2=np.array([sum(production_set_subs)/sum(c.production_set)*np.random.uniform(0.8,1.2)])                 
                  c.storage_TS=c.storage_TS*facteur_2
                  c.production_set=production_set_subs
          if (obtained_self_sufficiency < self_sufficiency) : #& (sum(c.storage_TS.sum(axis=1)<0)>0) :
              #store=np.random.choice(np.where(c.storage_TS.sum(axis=1)<0)[0],1)[0]
              #mutations=np.random.choice(np.where(importation>0)[0],min(np.random.randint(0,max(1,int(0.16*sum(importation>0)))),len(np.where(c.storage_TS[store]<=0)[0]),len(np.where(c.storage_TS[store]>=0)[0])),replace=False,p=importation[importation>0]/sum(importation[importation>0]))

              store=np.random.choice(n_store)
              mutations=np.random.choice(np.where(importation>0)[0],np.random.randint(0,max(1,int(0.16*sum(importation>0)))),replace=False,p=importation[importation>0]/sum(importation[importation>0]))
              import_level = importation[mutations]
              c.storage_TS[store][mutations]=c.storage_TS[store][mutations]+import_level*np.random.uniform(0,0.5,len(mutations))
         #     mutations_export = np.random.choice(np.where(importation<0)[0],min(np.random.randint(0,max(1,int(0.05*sum(importation<0)))),len(np.where(c.storage_TS[store]<=0)[0]),len(np.where(c.storage_TS[store]>=0)[0])),replace=False)
         #     if ((np.random.rand()<0.5)) :
         #         mutations_export_discharge = mutations_export[c.storage_TS[store][mutations_export]>0]
         #         alter_charge_export = np.random.choice(np.where(c.storage_TS[store]>=0)[0],size=len(mutations_export_discharge),replace=False) 
         #         changes_charge_export = np.min(np.reshape(np.concatenate((np.array(-importation[mutations_export_discharge]),c.storage_TS[store][mutations_export_discharge])),(-1,2)),axis=1)
         #         c.storage_TS[store][mutations_export_discharge]=c.storage_TS[store][mutations_export_discharge]-changes_charge_export
         #         c.storage_TS[store][alter_charge_export]=c.storage_TS[store][alter_charge_export]+changes_charge_export
         #     else : 
         #         mutations_export_charge = mutations_export[c.storage_TS[store][mutations_export]<0]
         #         alter_charge_export = np.random.choice(np.where(c.storage_TS[store]<=0)[0],size=len(mutations_export_charge),replace=False) 
         #         changes_charge_export = np.min(np.reshape(np.concatenate((np.array(-importation[mutations_export_charge]),c.storage_TS[store][mutations_export_charge])),(-1,2)),axis=1)
         #         c.storage_TS[store][mutations_export_charge]=c.storage_TS[store][mutations_export_charge]-changes_charge_export
         #         c.storage_TS[store][alter_charge_export]=c.storage_TS[store][alter_charge_export]+changes_charge_export
         ## ANNULé !! Supprime des exportations
         ##importation_2 = Non_movable_load-production-np.sum(c.storage_TS,axis=0)   
         ##     mutations_export_2 = np.random.choice(np.where(importation_2<0)[0],min(np.random.randint(0,max(1,int(0.2*sum(importation_2<0)),int(0.2*sum(importation_2>0)))),sum(importation_2<0)),replace=False)
              
         ##     if ((len(mutations_export_2)>0) & (len(np.where(importation_2>0)[0])>len(mutations_export_2)) ) :
         ##       mutations_import_2 = np.random.choice(np.where(importation_2>0)[0],len(mutations_export_2),replace=False)
              
         ##       store_2 = np.random.choice(np.where(np.array(c.storage_sum)>0)[0],1)[0]
         ##       goals_storage_TS = np.min(np.stack((-importation_2[mutations_export_2],importation_2[mutations_import_2],abs(c.storage_TS[store_2][mutations_export_2]),abs(c.storage_TS[store_2][mutations_import_2])),axis=1),axis=1)
         ##       mask_im=np.ones(len(mutations_export_2))
         ##       mask_ex=np.ones(len(mutations_import_2))
         ##       mask_im[c.storage_TS[store_2][mutations_import_2]>0]=storage_characteristics['Round-trip efficiency'][store_2]
         ##       mask_ex[c.storage_TS[store_2][mutations_export_2]>0]=storage_characteristics['Round-trip efficiency'][store_2]
         ##       c.storage_TS[store_2][mutations_export_2] = c.storage_TS[store_2][mutations_export_2]-goals_storage_TS*mask_ex
         ##       c.storage_TS[store_2][mutations_import_2] = c.storage_TS[store_2][mutations_import_2]+goals_storage_TS*mask_im
         ##       c.storage_sum[store_2]=-sum(c.storage_TS[store_2][c.storage_TS[store_2]<0])

          return(c)
    
def NON_JIT_mutation_contraintes_3(c, random_factors, choices,n_bits, Bounds_prod, Non_movable_load, constraint_num, constraint_level, prods_U,prod_C, n_store, n_contracts, time_resolution, storage_characteristics,Volums_prod,D_DSM_indexes,hyperparameters_operators_num):
       
        usage_ope = np.repeat(0, 25)
        #MUTATION DU CONTRAT
        
#        t0 = time.time()
        if ((random_factors[0]<hyperparameters_operators_num[0,0])) :  
              c=Eop.contract_operator(c,n_contracts)
              #c.contract=np.random.randint(0,n_contracts,1)[0]
              usage_ope[0]=1
              
#        t1=time.time()
        ## Semi-driven operator
        ## DIMINUTION DE LA PUISSANCE DU CONTRAT
        if ((random_factors[1]<hyperparameters_operators_num[0,0]) & np.any(c.trades>0)) :
            c=Eop.power_contract_operator(c,choices)
            #store = choices[0]
            #trades_pos = np.where(c.trades>0,c.trades,0)
            #if (max(trades_pos)<1e30):
            #    changes = np.random.choice(len(c.storage_TS[0]),np.random.randint(max(2,int(len(c.storage_TS[0])/hyperparameters_operators_num[1,0]))),p=trades_pos**10/sum(trades_pos**10))
            #    c.storage_TS[store][changes] = c.storage_TS[store][changes]+c.trades[changes]*np.random.uniform(0,0.2)        
            usage_ope[1]=1

      
        #Mutation de la production
#        t2=time.time()
#        if (random_factors[2]<0) :
#              new_set = np.array([min(Bounds_prod[i],max(0,c.production_set[i]+np.random.randint(min(-1,-int(Bounds_prod[i]/hyperparameters_operators['Production'][1])),max(1,int(Bounds_prod[i]/hyperparameters_operators['Production'][1])),1)[0])) for i in range(len(Bounds_prod))])
      ###        new_set = np.minimum(Bounds_prod,np.maximum(0,c.production_set+np.random.randint(np.minimum(-1,-(Bounds_prod/50)),np.maximum(1,(Bounds_prod/50)),len(Bounds_prod))) )
      ###        added_prod = np.dot((new_set - c.production_set),prods_U)##*np.random.normal(1,0.05,len(Non_movable_load))
#              added_prod = np.dot(new_set-c.production_set,Volums_prod) *np.random.uniform(0,1,1)/1000    
#              c.production_set=new_set
      ###        added_prod = np.random.rand(0,1,len(Non_movable_load)*n_store)
               
              ##repartition=np.random.rand(n_store*len(Non_movable_load)).reshape(n_store,len(Non_movable_load))
              ##c.storage_TS = c.storage_TS-added_prod*repartition/np.sum(repartition)
#              store = choices[14]
#              places = np.where(c.storage_TS[store]<0)[0]
#              repartition=np.random.rand(n_store*len(places)).reshape(n_store,len(places))
#              c.storage_TS[store,places] = c.storage_TS[store,places]-added_prod*repartition/np.sum(repartition)
              
#              usage_ope[2]=1
              
        if (random_factors[2]<hyperparameters_operators_num[0,1]) :
             #productor = np.random.choice(len(c.production_set),1)[0]
             #modifier=max(0,min(Bounds_prod[productor],c.production_set[productor]+np.random.randint(min(-1,-int(Bounds_prod[productor]/hyperparameters_operators_num[1,1])),max(1,int(Bounds_prod[productor]/hyperparameters_operators_num[1,1])),1)))
             #added_prod = (modifier-c.production_set[productor])*prods_U[productor]*np.random.uniform(0,1,1)*np.random.normal(1,0.2,n_bits)/1000    
             #c.production_set[productor]=modifier              
             #places = np.where(c.trades>0)[0]
             #store = choices[1]
             #c.storage_TS[store,places] = c.storage_TS[store,places]-added_prod[places]             
             c=Eop.production_operator(c,choices,Bounds_prod,prods_U,n_bits,hyperparameters_operators_num)
             
             usage_ope[2]=1
             
             
        if (random_factors[3]<hyperparameters_operators_num[0,1]/2) :
                  #productors = np.random.choice(len(c.production_set),2,replace=False)
                  #c.production_set[productors[0]]=max(min(c.production_set[productors[0]]-1,Bounds_prod[productors[0]]),0)
                  #c.production_set[productors[1]]=max(min(c.production_set[productors[1]]+1,Bounds_prod[productors[1]]),0)      
                  c=Eop.production_switch_operator(c,Bounds_prod)
                  usage_ope[3]=1


      ###        c.storage_TS = c.storage_TS-np.outer(added_prod,repartition).T/sum(repartition)/1000
              #facteur=np.random.rand(1)+0.5
              
              #for i in range(n_store):
              #    c.storage_TS[i]=c.storage_TS[i]*facteur
    # Mutation aléatoire de la série temporelle
              #mutations=[ np.random.choice(range(len(c.storage_TS[i])), size=np.random.randint(1,int(len(c.storage_TS[i])/200)+2), replace=False) for i in range(n_store)]
              #for store_mutations in range(len(mutations)) :
              #    c.storage_TS[store_mutations][mutations[store_mutations]]=c.storage_TS[store_mutations][mutations[store_mutations]]*np.random.uniform(0.6,1.4)
      ###        direction = np.random.randint(n_store)
      ###        sub_storage_TS = [np.where(c.storage_TS[i]<0)[0] if direction==0 else np.where(c.storage_TS[i]>0)[0] for i in range(n_store)]
      ###        mutations=[[] if len(sub_storage_TS[i])<5 else np.random.choice(sub_storage_TS[i], size=np.random.randint(1,int(len(sub_storage_TS[i])/200)+2), replace=False) for i in range(n_store)]
              
      ###        for store_mutations in range(len(mutations)) :
      ###            random_index=np.random.random(len(mutations[store_mutations]))
      ###            c.storage_TS[store_mutations][mutations[store_mutations]]=random_index/sum(random_index)*sum(c.storage_TS[store_mutations][mutations[store_mutations]])
        
        #MUTATIONS DES SERIES TEMPORELLES
        #RANDOM OPERATORS
        
        # Mutation aléatoire de la série temporelle
#        t3=time.time()
        if ((random_factors[4]<hyperparameters_operators_num[0,6])) :
                  #store = choices[2] 
                  #mutations=np.random.choice(range(len(c.storage_TS[store])), size=np.random.randint(1,int(len(c.storage_TS[store])/hyperparameters_operators_num[1,6])+4), replace=False)
                  #c.storage_TS[store][mutations]=(c.storage_TS[store][mutations]+np.random.normal(0,50*hyperparameters_operators_num[5,6],len(mutations)))*np.random.normal(1,hyperparameters_operators_num[5,6],len(mutations))
                  c=Eop.timeserie_operator(c,choices,hyperparameters_operators_num)
                  usage_ope[4]=1
     
        # Mutation aléatoire de la série temporelle sur des séquences voisines en respectant les sens
#        t4=time.time()
        if ((random_factors[5]<hyperparameters_operators_num[0,6])) :
                  #store = choices[3]     
                  #len_subset = np.random.randint(1,min(30,max(3,int(len(c.storage_TS[0])/hyperparameters_operators_num[1,6]))))                  
                  #starts = np.random.choice(range(0,len(c.storage_TS[0])-len_subset),np.random.randint(1,min(30,int(len(c.storage_TS[0])/(2*hyperparameters_operators_num[1,6]))+2)))              
                  #places = np.linspace(starts,starts+len_subset-1,len_subset).astype('int').T
                  #matrice_mult = np.random.normal(1,hyperparameters_operators_num[5,6],places.size).reshape(places.shape[0],places.shape[1])
                  #c.storage_TS[store,places] = c.storage_TS[store,places]*matrice_mult
                  c=Eop.timeserie_sequences_operator(c,choices,hyperparameters_operators_num)
                  usage_ope[5]=1
     
        # Mutation aléatoire de la série temporelle sur des séquences voisines
#        if ((random_factors[4]<0.0)) :
#                  store = choices[2]     
#                  len_subset = np.random.randint(1,min(30,max(3,int(len(c.storage_TS[0])/600))))                  
#                  starts = np.random.choice(range(0,len(c.storage_TS[0])-len_subset),np.random.randint(1,min(30,int(len(c.storage_TS[0])/600)+2)))              
#                  places = np.linspace(starts,starts+len_subset-1,len_subset).astype('int').T
#                  subset = c.storage_TS[store,places]
#                  sums = np.sum(subset,axis=1)
#                  random_index=np.random.random(places.size).reshape(places.shape[0],places.shape[1])-0.1
#                  vec_places = places.flatten()
#                  matrice_replacement = (sums*np.divide(random_index.T,np.sum(random_index,axis=1))).T
#                  c.storage_TS[store,places] = random_factors[30]*c.storage_TS[store,places]+(1-random_factors[30])*matrice_replacement
        
        ##MODIFICATION DE L'UTILISATION GLOBALE DES STOCKAGES
#        t5=time.time()
        if ((random_factors[6]<hyperparameters_operators_num[0,3])) :
              #store = choices[4]
              #c.storage_TS[store,] = c.storage_TS[store,]*np.random.uniform(1-1/hyperparameters_operators_num[6,3],1+1/hyperparameters_operators_num[6,3],1)
              c=Eop.storage_use_global_operator(c,choices,n_store,hyperparameters_operators_num)
              usage_ope[6]=1
              
        ##MODIFICATION DE L'UTILISATION DES STOCKAGES SUR DES SOUS-ENSEMBLES
#        t5=time.time()
        if (random_factors[7]<(hyperparameters_operators_num[0,3])):
               #store = choices[5]
               #width = np.random.randint(1,40)
               #len_subset = np.random.randint(1,max(3,int(len(c.storage_TS[0])/hyperparameters_operators_num[1,3])))              
               #starts = np.random.choice(range(0,len(c.storage_TS[0])-len_subset),width)              
               #vec_places = np.linspace(starts,starts+len_subset-1,len_subset).astype('int').T.flatten()
               #vec_replacement = c.storage_TS[store,vec_places]*np.random.uniform(1-1/hyperparameters_operators_num[6,3],1+1/hyperparameters_operators_num[6,3],1)
               #c.storage_TS[store,vec_places] = vec_replacement
               c=Eop.storage_use_local_operator(c,choices,hyperparameters_operators_num)
               usage_ope[7]=1
               
               #On introduit un transfert éventuel entre stockages
#        t6=time.time()
        if ((n_store>1) & (random_factors[8]<hyperparameters_operators_num[0,8]) ) :
                #(store_in,store_out)=np.random.choice(n_store,2,replace=False)
                #if (sum(c.storage_sum[np.ix_((store_in,store_out))])>0):
                #         switch_indexes1=np.random.choice(len(c.storage_TS[0]) ,np.random.randint(2,max(3,int(len(c.storage_TS[0])/(hyperparameters_operators_num[1,8]/4)))),replace=False)
                #         c.storage_TS[store_in,switch_indexes1]=np.sum(c.storage_TS[:,switch_indexes1],axis=0)*random_factors[21]*c.storage_sum[store_in]/sum(c.storage_sum[np.ix_((store_in,store_out))])
                #         c.storage_TS[store_out,switch_indexes1]=np.sum(c.storage_TS[:,switch_indexes1],axis=0)*(1-random_factors[21]*c.storage_sum[store_in]/sum(c.storage_sum[np.ix_((store_in,store_out))]))
                         c=Eop.storage_transfer_1_operator(c,n_store,random_factors,hyperparameters_operators_num)
                         usage_ope[8]=1
                         
                #On introduit un transfert éventuel entre stockages_v2
#        t7=time.time()    
        if ((n_store>1) & (random_factors[9]<hyperparameters_operators_num[0,8])) :
                 #(store_in,store_out)=np.random.choice(n_store,2,replace=False)
                 #len_subset = np.random.randint(1,min(30,max(3,int(len(c.storage_TS[0])/hyperparameters_operators_num[1,8]))))                  
                 #starts = np.random.choice(range(0,len(c.storage_TS[0])-len_subset),np.random.randint(1,min(30,int(len(c.storage_TS[0])/(hyperparameters_operators_num[1,8]/2))+2)))              
                 #places = np.linspace(starts,starts+len_subset-1,len_subset).astype('int').T
                 #subset = c.storage_TS[store_out,places]
                 #direction = -1
                 #subset_cha = np.where(np.sign(subset)==direction,subset,0)
                 #switch_cha = -abs(np.random.normal(hyperparameters_operators_num[3,8],hyperparameters_operators_num[5,8],places.size).reshape(len(starts),len_subset))*subset_cha
                 #c.storage_TS[store_in,places] = c.storage_TS[store_in,places]-switch_cha
                 #c.storage_TS[store_out,places] = c.storage_TS[store_out,places]+switch_cha

                 #direction = 1
                 #subset_dis = np.where(np.sign(subset)==direction,subset,0)
                 #switch_dis = abs(np.random.normal(hyperparameters_operators_num[3,8],hyperparameters_operators_num[5,8],places.size).reshape(len(starts),len_subset))*subset_dis
                 #c.storage_TS[store_in,places] = c.storage_TS[store_in,places]+switch_dis*storage_characteristics[4,store_out]/storage_characteristics[4,store_in]
                 #c.storage_TS[store_out,places] = c.storage_TS[store_out,places]-switch_dis
                 c=Eop.storage_transfer_2_operator(c,n_store,storage_characteristics,hyperparameters_operators_num)
                 usage_ope[9]=1

                #On introduit un transfert éventuel entre stockages_v3
 #       if ((n_store>1) & (random_factors[9]<0.05)) :
 #                (store_in,store_out)=np.random.choice(n_store,2,replace=False)
 #                len_subset = 144              
 #                starts = np.random.choice(range(0,len(c.storage_TS[0])-len_subset),np.random.randint(1,min(6,int(len(c.storage_TS[0])/600)+2)))              
 #                places = np.linspace(starts,starts+len_subset-1,len_subset).astype('int').T
 #                subset = c.storage_TS[store_out,places]
 #                coeff = np.random.uniform(0,1,1)
 #                switch_cha = coeff*subset
 #                c.storage_TS[store_in,places] = c.storage_TS[store_in,places]+switch_cha*storage_characteristics['Round-trip efficiency'][store_out]/storage_characteristics['Round-trip efficiency'][store_in]
 #                c.storage_TS[store_out,places] = c.storage_TS[store_out,places]-switch_cha
        
        
        ##SEMI-ORIENTED OPERATORS
        ## DIMINUTION DU VOLUME D'UN STOCKAGE ALEATOIRE
#        t8=time.time()   
        if ((random_factors[10]<hyperparameters_operators_num[0,2])& (sum(c.storage_TS[choices[6]])!=0)) :
            c=Eop.storage_volume_operator(c,choices,storage_characteristics,hyperparameters_operators_num)
            #store = choices[6]
            #losses = np.where(c.storage_TS[store]/storage_characteristics[4,store]-c.storage_TS[store]>0,c.storage_TS[store]/storage_characteristics[4,store]-c.storage_TS[store],0)
            #sum_diff_storages = np.cumsum(c.storage_TS[store]+losses) 
            #points = np.argmin(sum_diff_storages),np.argmax(sum_diff_storages)
            #if (points[0]!=points[1]):
            #    sub_storage = [np.fromiter(chain(range(min(points)),range(max(points),len(c.storage_TS[0]))),'int'),range(min(points),max(points))]
            #    changes = [np.random.choice(sub_storage[0],np.random.randint(max(2,int(len(sub_storage[0])/hyperparameters_operators_num[1,2]))),replace=False),np.random.choice(sub_storage[1],np.random.randint(1,max(2,int(len(sub_storage[1])/hyperparameters_operators_num[1,2]))),replace=False)]
            #    shift = np.random.uniform(0,np.ptp(sum_diff_storages)/hyperparameters_operators_num[6,2])

            #    if points[0]==min(points):
            #        c.storage_TS[store][changes[0]]=c.storage_TS[store][changes[0]]+shift*np.random.normal(hyperparameters_operators_num[3,2],hyperparameters_operators_num[5,2],len(changes[0]))/len(changes[0])
            #        c.storage_TS[store][changes[1]]=c.storage_TS[store][changes[1]]-shift*np.random.normal(hyperparameters_operators_num[3,2],hyperparameters_operators_num[5,2],len(changes[1]))/len(changes[1])
            #    else :
            #        c.storage_TS[store][changes[0]]=c.storage_TS[store][changes[0]]-shift*np.random.normal(hyperparameters_operators_num[3,2],hyperparameters_operators_num[5,2],len(changes[0]))/len(changes[0])
            #        c.storage_TS[store][changes[1]]=c.storage_TS[store][changes[1]]+shift*np.random.normal(hyperparameters_operators_num[3,2],hyperparameters_operators_num[5,2],len(changes[1]))/len(changes[1])
            usage_ope[10]=1


## MAXIMISATION DE L'UTILISATION DU VOLUME D'UN STOCKAGE ALEATOIRE
#        t8=time.time()   
#        if ((random_factors[10]<hyperparameters_operators['Storage volume'][0])& (sum(c.storage_TS[choices[5]])!=0)) :
#            store = choices[5]
#            losses = np.where(c.storage_TS[store]/storage_characteristics['Round-trip efficiency'][store]-c.storage_TS[store]>0,c.storage_TS[store]/storage_characteristics['Round-trip efficiency'][store]-c.storage_TS[store],0)
#            sum_diff_storages = np.cumsum(c.storage_TS[store]+losses) 
#            table_day = sum_diff_storages.reshape(int(time_resolution*24),int(len(sum_diff_storages)/(time_resolution*24)))
#            points = np.where(sum_diff_storages==min(np.max(table_day,axis=1)))[0][0],np.where(sum_diff_storages==max(np.min(table_day,axis=1)))[0][0]
#            if (points[0]!=points[1]):
#                sub_storage = [np.fromiter(chain(range(min(points)),range(max(points),len(c.storage_TS[0]))),'int'),range(min(points),max(points))]
#                changes = [np.random.choice(sub_storage[0],np.random.randint(max(2,int(len(sub_storage[0])/hyperparameters_operators['Storage volume'][1]))),replace=False),np.random.choice(sub_storage[1],np.random.randint(1,max(2,int(len(sub_storage[1])/hyperparameters_operators['Storage volume'][1]))),replace=False)]
#                shift = np.random.uniform(0,np.ptp(sum_diff_storages)/hyperparameters_operators['Storage volume'][6])

#                if points[0]==min(points):
#                    c.storage_TS[store][changes[0]]=c.storage_TS[store][changes[0]]+shift*np.random.normal(hyperparameters_operators['Storage volume'][3],hyperparameters_operators['Storage volume'][5],len(changes[0]))/len(changes[0])
#                    c.storage_TS[store][changes[1]]=c.storage_TS[store][changes[1]]-shift*np.random.normal(hyperparameters_operators['Storage volume'][3],hyperparameters_operators['Storage volume'][5],len(changes[1]))/len(changes[1])
#                else :
#                    c.storage_TS[store][changes[0]]=c.storage_TS[store][changes[0]]-shift*np.random.normal(hyperparameters_operators['Storage volume'][3],hyperparameters_operators['Storage volume'][5],len(changes[0]))/len(changes[0])
#                    c.storage_TS[store][changes[1]]=c.storage_TS[store][changes[1]]+shift*np.random.normal(hyperparameters_operators['Storage volume'][3],hyperparameters_operators['Storage volume'][5],len(changes[1]))/len(changes[1])
                


        ## DIMINUTION DE LA PUISSANCE D'UN STOCKAGE ALEATOIRE
#        t9=time.time()   
        if (random_factors[11]<(hyperparameters_operators_num[0,4])) :
            
            #store = choices[7]
            #abs_store=abs(c.storage_TS[store])
            #if (((sum(c.storage_TS[store]!=0)/hyperparameters_operators_num[1,4])>1) & (max(abs_store)<1e30)):
            #        changes = np.random.choice(len(c.storage_TS[store]),np.random.randint(max(2,int(len(c.storage_TS[store])/hyperparameters_operators_num[1,4]))),p=abs_store**10/sum(abs_store**10),replace=False)
            #        c.storage_TS[store][changes] = c.storage_TS[store][changes]*np.random.uniform(1-1/hyperparameters_operators_num[6,4],1)
            c=Eop.storage_power_operator(c,choices,hyperparameters_operators_num)        
            usage_ope[11]=1

            #sub_storage = np.random.choice(len(Non_movable_load),np.random.randint(2,max(3,int(len(Non_movable_load)/600))),replace=False)
            #c.storage_TS[store,sub_storage] = np.random.normal(np.mean(c.storage_TS[store,sub_storage]),np.std(c.storage_TS[store,sub_storage]),len(sub_storage))
       
        ##ANNULATION DES MOUVEMENTS OPPOSES DE 2 STOCKAGES ALEATOIRES
#        t10=time.time()   
        if (n_store>1) & (random_factors[12]<hyperparameters_operators_num[0,7]):
              c=Eop.opposite_moves_operator(c,n_store,hyperparameters_operators_num)
              #stores = np.random.choice (2,n_store,replace=False)
              #poss_changes = np.where(c.storage_TS[stores[0]]*c.storage_TS[stores[1]]<0)[0]
              #if (len(poss_changes)>1):
              #    changes = np.random.choice(poss_changes, max(2,int(len(poss_changes)/hyperparameters_operators_num[1,7])),replace=False)
              #    c.storage_TS[np.ix_(stores,changes)]=np.stack((np.sum(c.storage_TS[:,changes][stores],axis=0),np.repeat(0,len(changes))))
              usage_ope[12]=1
        ##ANNULATION DES DECHARGES LORS D'EXPORTATIONS (OU INVERSE)
      ##  if (random_factors[14]<hyperparameters_operators['Scheduling consistency'][0]):
      ##        store = choices[8]
       ###       poss_changes = np.where(c.storage_TS[store]*(c.trades+np.sum(c.storage_TS,axis=0))<0)[0]
      ##        poss_changes = np.where((c.storage_TS[store]>0) & (c.trades<0))[0]
      ##        if (len(poss_changes)>1):
      ##            changes = np.random.choice(poss_changes, np.random.randint(1,max(2,int(len(poss_changes)/hyperparameters_operators['Scheduling consistency'][1]))),replace=False)
      ##            c.storage_TS[store,changes]=c.storage_TS[store,changes]-np.minimum(c.storage_TS[store,changes],-c.trades[changes])*np.random.uniform(1-1/hyperparameters_operators['Scheduling consistency'][6],1)
         
    ### ANNULATION DES DECHARGES/EXPORT ou CHARGES/IMPORTS
#        t11=time.time()   
        if (random_factors[13]<hyperparameters_operators_num[0,5]):
             c=Eop.Scheduling_consistency_operator(c,choices,hyperparameters_operators_num)
             #store = choices[8]
             #len_subset = np.random.randint(1,min(100,max(3,int(len(c.storage_TS[0])/hyperparameters_operators_num[1,5]))))
             #place = np.random.choice(range(0,len(c.storage_TS[0])-len_subset))
             #subset = c.storage_TS[store,range(place,place+len_subset+1)]
             #bornes = c.trades[range(place,place+len_subset+1)]
             #energy_residuals = np.where(subset*bornes<0,np.sign(bornes)*np.minimum(abs(subset),abs(bornes)),0)
             #if ((energy_residuals!=0).any() & (energy_residuals==0).any()):
             #    c.storage_TS[store,range(place,place+len_subset+1)]=np.where(subset*bornes<0,subset+energy_residuals,(subset-sum(energy_residuals)/sum(energy_residuals==0)))
             usage_ope[13]=1    

        ## Long-term consistency
#        t12=time.time()   
        if ((random_factors[14]<hyperparameters_operators_num[0,12]) & (len(c.storage_TS[0])>(100*int(time_resolution*24)))):
            c=Eop.Long_term_consistency_operator(c,choices,random_factors,time_resolution,hyperparameters_operators_num)
            #store = choices[9]
            #time_span = int(time_resolution*24)
            #len_subset = np.random.randint(1,min(30,max(3,int(len(c.storage_TS[0])/time_span))))      
            #starts = np.random.choice(range(4*time_span,len(c.storage_TS[0])-time_span*(len_subset+4)),max(1,int(len(c.storage_TS[store]/hyperparameters_operators_num[6,12]))))
            #places = np.linspace(starts,starts+(len_subset-1)*time_span,len_subset).astype('int').T
            #places_mva = np.linspace(starts-2*time_span,starts+(len_subset+2-1)*time_span,len_subset+4).astype('int').T
            #subset = c.storage_TS[store,places_mva]
            #vec_places = places.flatten()
            #cumsum_mat = np.cumsum(subset,axis=1) 
            #window_width = 4
            #ma_vec = (cumsum_mat[:,window_width:] - cumsum_mat[:,:-window_width]) / window_width 
            #replacement = random_factors[22]*c.storage_TS[store,places]+(1-random_factors[22])*ma_vec
            #c.storage_TS[store,places] = replacement
            usage_ope[14]=1
            
            ###Interdaily consistency
        if ((random_factors[15]<hyperparameters_operators_num[0,12]) & (len(c.storage_TS[0])>(20*int(time_resolution*24)))):
            
            #n_days = int(len(Non_movable_load)/(24*time_resolution))
            #store=choices[10]
            #len_subset = int(time_resolution*24)
            #starts_ref = np.random.choice(range(n_days-1),1)[0],0,np.random.randint(1,n_days),np.random.choice((-1,1))
            #starts = np.arange(starts_ref[0],starts_ref[0]+starts_ref[2]+1)[::(-1)*starts_ref[3]]%n_days            
            #places = np.linspace(starts*int(time_resolution*24)+starts_ref[1],starts*int(time_resolution*24)+starts_ref[1]+(len_subset-1),len_subset).astype('int').T
            #distances = abs(starts-starts[0])
            #c.storage_TS[store,places] = np.multiply((1-(random_factors[23])*(1-distances/max(distances)/2)),c.storage_TS[store,places].T).T+np.outer(random_factors[23]*(1-distances/max(distances)/2),c.storage_TS[store,places[0]])
            c=Eop.Interdaily_consistency_operator(c,choices,random_factors,n_bits,time_resolution)
            usage_ope[15]=1
        
        ##APPLATISSEMENT DES COURBES
#        t13=time.time()   
        if (random_factors[16]<hyperparameters_operators_num[0,10]):
               #store = choices[11]
               #len_subset = np.random.randint(1,min(30,max(3,int(len(c.storage_TS[0])/hyperparameters_operators_num[1,10]))))              
               #starts = np.random.choice(range(12,len(c.storage_TS[0])-len_subset-12),np.random.randint(1,min(30,int(len(c.storage_TS[0])/hyperparameters_operators_num[6,10])+2)))              
               #places = np.linspace(starts,starts+len_subset-1,len_subset).astype('int').T
               #places_mva = np.linspace(starts-6,starts+len_subset-1+6,len_subset+12).astype('int').T
               #subset = c.storage_TS[store,places_mva]  
               #vec_places = places.flatten()
               #cumsum_mat = np.cumsum(subset,axis=1) 
               #window_width = 12
               #ma_vec = (cumsum_mat[:,window_width:] - cumsum_mat[:,:-window_width]) / window_width 

               #vec_replacement = (random_factors[24]*c.storage_TS[store,places].flatten()+(1-random_factors[24])*ma_vec.flatten())          
               
               #c.storage_TS[store,vec_places] = vec_replacement
               c=Eop.curve_smoothing_operator(c,choices,random_factors,hyperparameters_operators_num)
               usage_ope[16]=1
               
###           bornes = np.sign(c.storage_TS[store,changes])*np.minimum(abs(c.storage_TS[store,changes]),abs(c.trades[changes]))
       ###           c.storage_TS[store,changes]=c.storage_TS[store,changes]-bornes*np.random.uniform(0,1,len(changes))


                      
        #Specification des rôles des stockages
#        if ((random_factors[18]<0.05) & n_store>1):
#            (store_longterm,store_shortterm)=np.random.choice(n_store,2,replace=False)
#            len_subset = 144              
#            starts = np.random.choice(range(0,len(c.storage_TS[0])-len_subset),np.random.randint(1,min(20,int(len(c.storage_TS[0])/600)+2)))              
#            places = np.linspace(starts,starts+len_subset-1,len_subset).astype('int').T
#            subset = c.storage_TS[store_shortterm,places]
#            subset_losses_included = np.where(subset>0,subset/storage_characteristics['Round-trip efficiency'][store_shortterm],subset)
#            sums = np.sum(subset_losses_included,axis=1)
#            coeff = np.random.uniform(0,1,1)
#            switch_cha = coeff*sums
#            c.storage_TS[store_longterm,places] = c.storage_TS[store_longterm,places]-np.repeat(switch_cha/144,144).reshape(len(starts),144)
#            c.storage_TS[store_shortterm,places] = c.storage_TS[store_shortterm,places]+np.repeat(switch_cha/144,144).reshape(len(starts),144)

        #Specification des rôles des stockages
#        t14=time.time()   
        if  ((random_factors[17]<hyperparameters_operators_num[0,9]) & n_store>1):
            #(store_longterm,store_shortterm)=np.random.choice(n_store,2,replace=False)
            #len_subset = np.random.randint(1,max(3,int(len(c.storage_TS[0])/hyperparameters_operators_num[1,9])))         
            #starts = np.random.choice(range(0,len(c.storage_TS[0])-len_subset),np.random.randint(1,min(20,int(len(c.storage_TS[0])/hyperparameters_operators_num[6,9])+2)))              
            #places = np.linspace(starts,starts+len_subset-1,len_subset).astype('int').T
            #subset = c.storage_TS[:,places][(store_longterm,store_shortterm),:,:]
            
            #total_actions = np.sum(subset,axis=0)
            #results = np.sum(total_actions,axis=1)
            #trend = results/len_subset
            #noise = total_actions-np.repeat(trend,len_subset).reshape(len(starts),len_subset)
            #coeff = random_factors[25]
            #c.storage_TS[store_longterm,places] = (1-coeff)*c.storage_TS[store_longterm,places]+np.repeat(coeff*trend,len_subset).reshape(len(starts),len_subset)
            #c.storage_TS[store_shortterm,places] = (1-coeff)*c.storage_TS[store_shortterm,places]+coeff*noise
            c=Eop.storage_specification_operator(c,random_factors,n_store,storage_characteristics,hyperparameters_operators_num)
            usage_ope[17]=1
   
        
        
        ##MODIFICATION D'UN SOUS-ENSEMBLE ALEATOIRE DES STOCKAGES
       # if ((random_factors[2]<hyperparameters_operators['Storage timeserie'][0])) :
       #       sub_storage = np.random.choice(len(Non_movable_load),np.random.randint(0,max(2,int(len(Non_movable_load)/hyperparameters_operators['Storage timeserie'][1]))),replace=False)
       #       c.storage_TS[:,sub_storage] = np.multiply(np.random.normal(np.random.uniform(hyperparameters_operators['Storage timeserie'][2],hyperparameters_operators['Storage timeserie'][3],1),np.random.uniform(hyperparameters_operators['Storage timeserie'][4],hyperparameters_operators['Storage timeserie'][5],1),len(sub_storage)),c.storage_TS[:,sub_storage])
        
        ##MODIFICATION D'UN SOUS-ENSEMBLE ALEATOIRE DES STOCKAGES v2
       # if ((random_factors[2]<0.03)) :
       #       store = choices[4]
       #       sub_storage = np.random.choice(len(Non_movable_load),np.random.randint(0,max(2,int(len(Non_movable_load)/hyperparameters_operators['Storage timeserie'][1]))),replace=False)
       #       random_index=np.random.random(len(sub_storage))
       #       c.storage_TS[store,sub_storage]=random_index/sum(random_index)*sum(c.storage_TS[store,sub_storage])
           
        
        ##RE-ORDONNATION ALEATOIRE D'UN SOUS-ENSEMBLE D'UN STOCKAGE
#        if ((random_factors[11]<hyperparameters_operators['Storage rescheduling'][0])) & (sum(c.storage_sum)>0) :
#              store = choices[2]
#              sub_storage = np.random.choice(len(Non_movable_load),np.random.randint(0,max(2,int(len(Non_movable_load)/hyperparameters_operators['Storage timeserie'][1]))),replace=False)
#              c.storage_TS[store,sub_storage] = c.storage_TS[store,sorted(sub_storage, key=lambda x: np.random.rand())]*np.random.normal(0,0.1,len(sub_storage))


     ###   if ((random_factors[7]<0.3)) :
     ###         sub_storage = np.random.choice(len(Non_movable_load),np.random.randint(0,max(2,int(len(Non_movable_load)/800))),replace=False)
     ###         c.storage_TS[:,sub_storage] = c.storage_TS[:,sorted(sub_storage, key=lambda x: np.random.rand())]
        
        
                  
        ##APPLATISSEMENT DES COURBES
#        if (random_factors[15]<0.1):
#              store = np.random.choice (n_store,1)[0]
       ###       poss_changes = np.where(c.storage_TS[store]*(c.trades+np.sum(c.storage_TS,axis=0))<0)[0]
#              positions = np.random.choice(range(1,len(c.storage_TS[store])-1),np.random.randint(1,len(c.storage_TS[store])/600),replace=False)
              
#              set_range = np.random.randint(0,min(25,min(positions),min(len(c.storage_TS[0])-positions).astype('int')))
#              if (set_range>0):
#                  indexes=[range(place-set_range,place+set_range+1) for place in positions]
#                  for k in indexes :
#                      c.storage_TS[store][k] = (random_factors[18]*c.storage_TS[store][k]+(1-random_factors[18])*np.mean(c.storage_TS[store][k]))*np.random.normal(1,0.1,2*set_range+1)


            ## OPERATEUR DE CONTRAINTE
        if (random_factors[18]<hyperparameters_operators_num[0,11]) :
            c=Eop.constraint_operator(c,constraint_num,hyperparameters_operators_num)
            usage_ope[18]=1
                
    ## REDUCTION DE L'EXPORTATION
      #  if (random_factors[16]<hyperparameters_operators['Exportation reducing'][0]) :
 #####       if (random_factors[16]<0.3) :
 #####           sample = c.trades<0
 #####           prob = c.trades[sample]
 #####           mutations_TS = np.random.choice(np.where(sample)[0],np.random.randint(0,max(2,int(sum(sample)/300))),p = prob/sum(prob),replace=False)
 #####           repartition_2=np.random.rand(n_store*len(mutations_TS)).reshape(len(mutations_TS),n_store)*c.storage_sum
 #####           repartition_2=np.divide(repartition_2.T,np.sum(repartition_2,axis=1))
 #####           c.storage_TS[:,mutations_TS] = c.storage_TS[:,mutations_TS]+np.random.uniform(0,c.trades[mutations_TS],len(mutations_TS))*repartition_2

 #####       if (random_factors[16]<0.3) :
 #####           mutations_export_2 = np.random.choice(np.where(c.trades<0)[0],np.random.randint(0,max(1,min(int(sum(c.trades<0)/700),int(sum(c.trades>0)/700)))),replace=False)
 #####           mutations_import_2 = np.random.choice(np.where(c.trades>0)[0],len(mutations_export_2),replace=False)
 #####           store=choices[3]      
  #####          shift = np.random.rand(len(mutations_export_2))*np.min(np.stack((-c.trades[mutations_export_2],c.trades[mutations_import_2]),axis=1),axis=1)
 #####           c.storage_TS[store,mutations_export_2]=c.storage_TS[store,mutations_export_2]-shift
 #####           c.storage_TS[store,mutations_import_2]=c.storage_TS[store,mutations_import_2]-shift
            
#####            mutations_export_2 = np.random.choice(np.where(c.trades<0)[0],np.random.randint(0,max(1,min(int(sum(c.trades<0)/800),int(sum(c.trades>0)/800)))),replace=False)
#####            mutations_import_2 = np.where(c.trades>0)[0][(np.searchsorted(np.where(c.trades>0)[0], mutations_export_2, side="right"))]
#####            store=choices[3]      
#####            shift = np.random.rand(len(mutations_export_2))*np.min(np.stack((-c.trades[mutations_export_2],c.trades[mutations_import_2]),axis=1),axis=1)
#####            c.storage_TS[store,mutations_export_2]=c.storage_TS[store,mutations_export_2]-shift
#####            c.storage_TS[store,mutations_import_2]=c.storage_TS[store,mutations_import_2]-shift


        #Mutation du DSM 
          #if ((np.random.rand()<0.01)) :
        if ((random_factors[19]<hyperparameters_operators_num[0,13])) :
              #mutations_Y_DSM=np.random.choice(len(c.Y_DSM), size=np.random.randint(1,max(2,int(len(c.Y_DSM)/hyperparameters_operators_num[1,13]))), replace=False)
                  # flip the bit         
              #c.Y_DSM[mutations_Y_DSM]= np.multiply(np.random.normal(1,np.random.uniform(hyperparameters_operators_num[4,13],hyperparameters_operators_num[5,13],1),len(mutations_Y_DSM)),c.Y_DSM[mutations_Y_DSM])
              c=Eop.Y_DSM_operator(c,hyperparameters_operators_num)
              usage_ope[19]=1
              
        if ((random_factors[30]<hyperparameters_operators_num[0,10])) :
              c=Eop.Y_DSM_smoothing_operator(c,random_factors,hyperparameters_operators_num)
              usage_ope[21]=1
              
        if ((random_factors[31]<hyperparameters_operators_num[0,12]) & (len(c.Y_DSM)>(20*int(time_resolution*24)))):
              c=Eop.Y_DSM_Interdaily_consistency_operator(c,random_factors,n_bits,time_resolution)
              usage_ope[22]=1              
        
              #c.Y_DSM[mutations_Y_DSM]=random_index*c.Y_DSM[mutations_Y_DSM]

           #   days_mutations_DSM = np.random.choice(len(c.D_DSM),int(len(c.D_DSM)/30))
           #   for day in days_mutations_DSM:
           #       mutations_D_DSM=np.random.choice(range(len(c.D_DSM[day,])), size=np.random.randint(1,int(len(c.D_DSM[day,])/20)+2), replace=False)
                      # flip the bit         
           #       random_index=np.random.random(len(mutations_D_DSM))
           #       c.D_DSM[day,mutations_D_DSM]=random_index/sum(random_index)*sum(c.D_DSM[day,mutations_D_DSM])
     ###         mutations_D_DSM=np.random.choice(range(len(c.D_DSM[0,])), size=np.random.randint(1,int(len(c.D_DSM[0,])/10)+2), replace=False)
     ###        random_index=np.random.random(len(mutations_D_DSM))
     ###         for day in range(len(c.D_DSM)):
     ###             if ((np.random.rand()<0.4)) :
     ###                 c.D_DSM[day,mutations_D_DSM]=random_index/sum(random_index)*sum(c.D_DSM[day,mutations_D_DSM])
        if ((random_factors[20]<hyperparameters_operators_num[0,14]) & (len(D_DSM_indexes)>0) & (c.D_DSM.shape[1]>1)) :
                c=Eop.D_DSM_operator(c,D_DSM_indexes,hyperparameters_operators_num)
                #mutations_D_DSM=np.random.choice(D_DSM_indexes, size=np.random.randint(1,max(2,int(len(D_DSM_indexes)/hyperparameters_operators_num[1,14]))), replace=False)
                 # flip the bit         
                #c.D_DSM[mutations_D_DSM]= np.multiply(np.random.normal(1,np.random.uniform(hyperparameters_operators_num[4,14],hyperparameters_operators_num[5,14],1),len(c.D_DSM[0])),c.D_DSM[mutations_D_DSM])
                usage_ope[20]=1 
                
        if ((random_factors[32]<hyperparameters_operators_num[0,10]) & (len(D_DSM_indexes)>0) & (c.D_DSM.shape[1]>1)) :
              c=Eop.D_DSM_smoothing_operator(c,D_DSM_indexes,random_factors,hyperparameters_operators_num)
              usage_ope[23]=1
              
        if ((random_factors[33]<hyperparameters_operators_num[0,12]) & (len(D_DSM_indexes)>50) & (c.D_DSM.shape[1]>1)):
              c=Eop.D_DSM_Interdaily_consistency_operator(c,D_DSM_indexes,random_factors,time_resolution)
              usage_ope[24]=1              
         

          
     ###                sums = c.storage_TS[store_in1][switch_indexes1]+ c.storage_TS[store_out1][switch_indexes1]
     ###                ratios = np.random.uniform(-0.2,1.2,len(switch_indexes1))
     ###                c.storage_TS[store_in1][switch_indexes1]=sums*ratios
     ###                c.storage_TS[store_out1][switch_indexes1]=sums*(1-ratios)
     ###         else :
                          
     ###             store_in1=np.random.choice(n_store)
     ###             store_out1=np.random.choice(n_store)
     ###             if (store_in1 != store_out1) :
     ###                switch_indexes1=np.random.choice(len(c.storage_TS[0]) ,max(2,int(len(c.storage_TS[0])/200)),replace=False)

     ###                c.storage_TS[store_in1][switch_indexes1]=c.storage_TS[store_in1][switch_indexes1]+ c.storage_TS[store_out1][switch_indexes1]
     ###               c.storage_TS[store_out1][switch_indexes1]=np.repeat(0,len(switch_indexes1))

     ###         production = np.sum(((np.dot(c.production_set,prods_U),prod_C)),axis=0)/1000
     ###         importation = Non_movable_load+np.concatenate(c.D_DSM)+c.Y_DSM-production-np.sum(c.storage_TS,axis=0)
     ###         obtained_self_sufficiency = 1-(np.sum(np.where(importation>0,importation,0))/(np.sum(Non_movable_load)+np.sum(c.Y_DSM)+np.sum(c.D_DSM)))

        #On créé un opérateur qui engendre ou annule des mouvements opposés d'un stockage
     ###         store_opp = np.random.choice(np.where(np.array([any(c.storage_TS[i]>0)  & any(c.storage_TS[i]<0) for i in range(len(c.storage_sum))]))[0],1)[0]
     ###         sub_storage_opposite = [np.random.choice(np.where(c.storage_TS[store_opp]<0)[0],np.random.randint(1,max(2,int(0.04*len(np.where(c.storage_TS[store_opp]<0)[0])))),replace=False),np.random.choice(np.where(c.storage_TS[store_opp]>0)[0],np.random.randint(1,max(2,int(0.04*len(np.where(c.storage_TS[store_opp]>0)[0])))),replace=False)] 
     ###         inflation= np.random.uniform(0.7,1.3) if obtained_self_sufficiency<self_sufficiency else np.random.uniform(0.4,1.05)
     ###         sens_limite=np.argmin([-sum(c.storage_TS[store_opp][sub_storage_opposite[0]]),sum(c.storage_TS[store_opp][sub_storage_opposite[1]])/storage_characteristics['Round-trip efficiency'][store_opp]])
     ###         creation = abs(sum(c.storage_TS[store_opp][sub_storage_opposite[sens_limite]]))*(inflation-1)
      ###        c.storage_TS[store_opp][sub_storage_opposite[sens_limite]]=c.storage_TS[store_opp][sub_storage_opposite[sens_limite]]*inflation
      ###        if (sens_limite==0):
      ###          c.storage_TS[store_opp][sub_storage_opposite[1]]=c.storage_TS[store_opp][sub_storage_opposite[1]]+c.storage_TS[store_opp][sub_storage_opposite[1]]/sum(c.storage_TS[store_opp][sub_storage_opposite[1]])*creation*storage_characteristics['Round-trip efficiency'][store_opp]
      ###        else :
      ###          c.storage_TS[store_opp][sub_storage_opposite[0]]=c.storage_TS[store_opp][sub_storage_opposite[0]]-c.storage_TS[store_opp][sub_storage_opposite[0]]/sum(c.storage_TS[store_opp][sub_storage_opposite[0]])*creation/storage_characteristics['Round-trip efficiency'][store_opp]
              #c.storage_sum[store_opp]=-sum(c.storage_TS[store_opp][c.storage_TS[store_opp]<0])
        
          
      ###        production_set_subs=np.array([min(Bounds_prod[i],max(0,c.production_set[i]+np.random.randint(0,max(2,int(Bounds_prod[i]/10)),1)[0])) for i in range(len(Bounds_prod))])
      ###        facteur_2=np.array([sum(production_set_subs)/sum(c.production_set)*np.random.uniform(0.8,1.2)])                 
      ###        c.storage_TS=c.storage_TS*facteur_2
      ###        c.production_set=production_set_subs
              #store=np.random.choice(np.where(c.storage_TS.sum(axis=1)<0)[0],1)[0]
              #mutations=np.random.choice(np.where(importation>0)[0],min(np.random.randint(0,max(1,int(0.16*sum(importation>0)))),len(np.where(c.storage_TS[store]<=0)[0]),len(np.where(c.storage_TS[store]>=0)[0])),replace=False,p=importation[importation>0]/sum(importation[importation>0]))

      ###        store=np.random.choice(n_store)
       ###       mutations=np.random.choice(np.where(importation>0)[0],np.random.randint(0,max(1,int(0.16*sum(importation>0)))),replace=False,p=importation[importation>0]/sum(importation[importation>0]))
      ###        import_level = importation[mutations]
      ###        c.storage_TS[store][mutations]=c.storage_TS[store][mutations]+import_level*np.random.uniform(0,0.5,len(mutations))
         #     mutations_export = np.random.choice(np.where(importation<0)[0],min(np.random.randint(0,max(1,int(0.05*sum(importation<0)))),len(np.where(c.storage_TS[store]<=0)[0]),len(np.where(c.storage_TS[store]>=0)[0])),replace=False)
         #     if ((np.random.rand()<0.5)) :
         #         mutations_export_discharge = mutations_export[c.storage_TS[store][mutations_export]>0]
         #         alter_charge_export = np.random.choice(np.where(c.storage_TS[store]>=0)[0],size=len(mutations_export_discharge),replace=False) 
         #         changes_charge_export = np.min(np.reshape(np.concatenate((np.array(-importation[mutations_export_discharge]),c.storage_TS[store][mutations_export_discharge])),(-1,2)),axis=1)
         #         c.storage_TS[store][mutations_export_discharge]=c.storage_TS[store][mutations_export_discharge]-changes_charge_export
         #         c.storage_TS[store][alter_charge_export]=c.storage_TS[store][alter_charge_export]+changes_charge_export
         #     else : 
         #         mutations_export_charge = mutations_export[c.storage_TS[store][mutations_export]<0]
         #         alter_charge_export = np.random.choice(np.where(c.storage_TS[store]<=0)[0],size=len(mutations_export_charge),replace=False) 
         #         changes_charge_export = np.min(np.reshape(np.concatenate((np.array(-importation[mutations_export_charge]),c.storage_TS[store][mutations_export_charge])),(-1,2)),axis=1)
         #         c.storage_TS[store][mutations_export_charge]=c.storage_TS[store][mutations_export_charge]-changes_charge_export
         #         c.storage_TS[store][alter_charge_export]=c.storage_TS[store][alter_charge_export]+changes_charge_export
         ## ANNULé !! Supprime des exportations
         ##importation_2 = Non_movable_load-production-np.sum(c.storage_TS,axis=0)   
         ##     mutations_export_2 = np.random.choice(np.where(importation_2<0)[0],min(np.random.randint(0,max(1,int(0.2*sum(importation_2<0)),int(0.2*sum(importation_2>0)))),sum(importation_2<0)),replace=False)
              
         ##     if ((len(mutations_export_2)>0) & (len(np.where(importation_2>0)[0])>len(mutations_export_2)) ) :
         ##       mutations_import_2 = np.random.choice(np.where(importation_2>0)[0],len(mutations_export_2),replace=False)
              
         ##       store_2 = np.random.choice(np.where(np.array(c.storage_sum)>0)[0],1)[0]
         ##       goals_storage_TS = np.min(np.stack((-importation_2[mutations_export_2],importation_2[mutations_import_2],abs(c.storage_TS[store_2][mutations_export_2]),abs(c.storage_TS[store_2][mutations_import_2])),axis=1),axis=1)
         ##       mask_im=np.ones(len(mutations_export_2))
         ##       mask_ex=np.ones(len(mutations_import_2))
         ##       mask_im[c.storage_TS[store_2][mutations_import_2]>0]=storage_characteristics['Round-trip efficiency'][store_2]
         ##       mask_ex[c.storage_TS[store_2][mutations_export_2]>0]=storage_characteristics['Round-trip efficiency'][store_2]
         ##       c.storage_TS[store_2][mutations_export_2] = c.storage_TS[store_2][mutations_export_2]-goals_storage_TS*mask_ex
         ##       c.storage_TS[store_2][mutations_import_2] = c.storage_TS[store_2][mutations_import_2]+goals_storage_TS*mask_im
         ##       c.storage_sum[store_2]=-sum(c.storage_TS[store_2][c.storage_TS[store_2]<0])
#        print("t1",t1-t0,"t2",t2-t1,"t3",t3-t2,"t4",t4-t3,"t5",t5-t4,"t6",t6-t5,"t7",t7-t6,"t8",t8-t7,"t9",t9-t8,"t10",t10-t9,"t11",t11-t10,"t12",t12-t11,"t13",t13-t12,"t14",t14-t13)
        return(c,usage_ope)
#        return(c)

#def bouclage_stockage(c,n_store,storage_characteristics):

####Correction des TS pour boucler le stockage
#    for i in range(n_store):
#        c.storage_sum[i]=-sum(c.storage_TS[i][c.storage_TS[i]<0])
#        c.storage_TS[i][c.storage_TS[i]>0]=c.storage_TS[i][c.storage_TS[i]>0]*c.storage_sum[i]*storage_characteristics['Round-trip efficiency'][i]/sum(c.storage_TS[i][c.storage_TS[i]>0])
#    return (c)

def bouclages_old(c,n_store,storage_characteristics,total_Y_movable_load,total_D_Movable_load,D_DSM_indexes ):

####Correction des TS pour boucler le stockage
    #On met tout à zéro s'il n'y a que des valeurs positives ou négatives.
    c.storage_TS = np.multiply(1-np.all(c.storage_TS<=0,axis=1),c.storage_TS.T).T
    c.storage_TS = np.multiply(1-np.all(c.storage_TS>=0,axis=1),c.storage_TS.T).T
    c.storage_sum=-np.sum(np.where(c.storage_TS<0,c.storage_TS,0),axis=1)
    
    coeffs = np.multiply(c.storage_sum,np.divide(storage_characteristics[4,:],1*(c.storage_sum==0)+np.sum(np.where(c.storage_TS>0,c.storage_TS,0),axis=1))) if np.sum(c.storage_TS>0)>0 else 1
    c.storage_TS=np.where(c.storage_TS>0,np.multiply(coeffs,c.storage_TS.T).T,c.storage_TS)
    
    if (total_Y_movable_load!=0 ) :        
        c.Y_DSM = c.Y_DSM*total_Y_movable_load/sum(c.Y_DSM)
             
    c.D_DSM[D_DSM_indexes[0],:] = np.multiply(c.D_DSM[D_DSM_indexes[0],:].T,total_D_Movable_load[D_DSM_indexes[0]]/np.sum(c.D_DSM[D_DSM_indexes[0],:],axis=1)).T
        
    return (c)

def non_JIT_bouclages(c,n_store,storage_characteristics,total_Y_movable_load,total_D_Movable_load,D_DSM_indexes ):

####Correction des TS pour boucler le stockage
    #On met tout à zéro s'il n'y a que des valeurs positives ou négatives.
    c.storage_TS = np.multiply(1-np.all(c.storage_TS<=0,axis=1),c.storage_TS.T).T
    c.storage_TS = np.multiply(1-np.all(c.storage_TS>=0,axis=1),c.storage_TS.T).T
    c.storage_sum=-np.sum(np.where(c.storage_TS<0,c.storage_TS,0),axis=1)
    absolutes = np.sum(abs(c.storage_TS),axis=1)
    volumes = absolutes-c.storage_sum*(1+np.array(storage_characteristics[4,:]))
    volumes = np.where(volumes>0,volumes/np.array(storage_characteristics[4,:]),volumes)
    ind=(volumes==0)
    
    if (sum(ind)!=n_store):
    
        ran = np.delete(np.arange(n_store),ind)
        subset =[np.where(np.sign(c.storage_TS[store])*np.sign(volumes[store])==-1)[0] for store in ran]

        lengths = np.array([len(x) for x in subset])

        subset2=np.array([x[range(min(lengths))] for x in subset])

        len_subset = np.random.randint(1,1+min(lengths))

        choices = np.random.choice(min(lengths),len_subset,replace=False)
        subset_reduced = np.array(subset2[:,choices])
        indexes = np.repeat(ran,len_subset),subset_reduced.flatten()

        coeffs = np.divide(abs(volumes[~ ind]),abs(np.sum(c.storage_TS[indexes].reshape(len(ran),len_subset),axis=1)))
     
        c.storage_TS[indexes]=c.storage_TS[indexes]*(np.repeat(1+coeffs,len_subset))
        c.storage_sum=-np.sum(np.where(c.storage_TS<0,c.storage_TS,0),axis=1)
    
    if (total_Y_movable_load!=0 ) :        
        c.Y_DSM = c.Y_DSM*total_Y_movable_load/sum(c.Y_DSM)
             
    c.D_DSM[D_DSM_indexes[0],:] = np.multiply(c.D_DSM[D_DSM_indexes[0],:].T,total_D_Movable_load[D_DSM_indexes[0]]/np.sum(c.D_DSM[D_DSM_indexes[0],:],axis=1)).T
        
    return (c)

@jit(nopython=True)
def bouclages_ol(c,n_store,storage_characteristics,total_Y_movable_load,total_D_Movable_load,D_DSM_indexes ):

####Correction des TS pour boucler le stockage
    #On met tout à zéro s'il n'y a que des valeurs positives ou négatives.
#    c.storage_TS = np.multiply(1-np.all(c.storage_TS<=0,axis=1),c.storage_TS.T).T
##    c.storage_TS = np.multiply(np.repeat(np.int64(1),len(c.storage_TS))-np.array([np.all(c.storage_TS[i]<=0) for i in range(n_store)],dtype=np.int64),c.storage_TS.T).T
##    x = np.array([(np.int64(1)-np.all(c.storage_TS[i]<=0)) *c.storage_TS[i] for i in range(n_store)])
    for i in range(n_store) : 
        if (np.all(c.storage_TS[i]<=0) | np.all(c.storage_TS[i]>=0)) :
            c.storage_TS[i]=0
    c.storage_sum=np.array([-np.sum(np.where(c.storage_TS[i]<0,c.storage_TS,0)) for i in range(n_store)])
    absolutes = np.array([np.sum(np.abs(c.storage_TS[i])) for i in range(n_store)])
    discharge_excess = absolutes-c.storage_sum*(1+storage_characteristics[4,:])
    #indexes = np.where(discharge_excess>0)[0]
    #for i in indexes :
    #    discharge_excess[i]=discharge_excess[i]/storage_characteristics[4,i] 

    ind=(discharge_excess==0)
    
    if (sum(ind)!=n_store):
    
        ran = np.delete(np.arange(n_store),np.where(ind)[0])
        subset =[np.where(np.sign(c.storage_TS[store])*np.sign(discharge_excess[store])==-1)[0] for store in ran]

        lengths = np.array([len(x) for x in subset])

        subset2 = np.zeros((len(ran),min(lengths)),dtype=np.int64)
        for i in range(len(ran)):
            subset2[i]=subset[i][np.arange(min(lengths))] 

        len_subset = np.random.randint(1,1+min(lengths))

        choices = np.random.choice(min(lengths),len_subset,replace=False)
        subset_reduced = subset2[:,choices]

        theoric_volumes = (np.abs(discharge_excess[~ ind]))
        effective_volumes=np.array([abs(sum(c.storage_TS[ran[i]][subset_reduced[i]]))for i in range(len(ran))])
        
        coeffs = np.divide(theoric_volumes,effective_volumes)
        
        for i in range(len(ran)) : 
            c.storage_TS[ran[i]][subset_reduced[i]]=c.storage_TS[ran[i]][subset_reduced[i]]*(np.repeat(1+coeffs[i],len_subset))
            c.storage_sum[ran[i]]=-np.sum(np.where(c.storage_TS[ran[i]]<0,c.storage_TS[ran[i]],0))
    
    if (total_Y_movable_load!=0 ) :        
        c.Y_DSM = c.Y_DSM*total_Y_movable_load/sum(c.Y_DSM)
             
    c.D_DSM[D_DSM_indexes,:] = np.multiply(c.D_DSM[D_DSM_indexes,:].T,total_D_Movable_load[D_DSM_indexes]/np.sum(c.D_DSM[D_DSM_indexes,:],axis=1)).T
        
    return (c)

@jit(nopython=True)
def bouclages(c,n_store,storage_characteristics,total_Y_movable_load,total_D_Movable_load,D_DSM_indexes ):

####Correction des TS pour boucler le stockage
    #On met tout à zéro s'il n'y a que des valeurs positives ou négatives.
    
    for i in range(n_store) : 
        if (np.all(c.storage_TS[i]<=0) | np.all(c.storage_TS[i]>=0)) :
            c.storage_TS[i]=0
    c.storage_sum=np.array([-np.sum(np.where(c.storage_TS[i]<0,c.storage_TS[i],0)) for i in range(n_store)])
    discharge_excess = np.array([np.sum(np.where(c.storage_TS[i]>0,c.storage_TS[i],0)) for i in range(n_store)])-c.storage_sum*storage_characteristics[4,:]
    discharge_excess = np.where(discharge_excess>0,discharge_excess/storage_characteristics[4,:],discharge_excess*storage_characteristics[4,:])
    ind=(discharge_excess==0)
    
    if (sum(ind)!=n_store):
    
        ran = np.delete(np.arange(n_store),np.where(ind)[0])
        directions = [np.sign(discharge_excess[store]) for store in range(n_store)]
        subset =[np.where(np.sign(c.storage_TS[store])*directions[store]==-1)[0] for store in ran]
        lengths = np.array([len(x) for x in subset])
        len_reduction = np.random.randint(1,1+min(lengths))        
        choices = np.random.choice(min(lengths),len_reduction,replace=False)
        subset_reduced = [subset[i][choices] for i in range(len(ran))]

        coeffs = np.array([abs(discharge_excess[ran[i]])/(storage_characteristics[4,ran[i]] if directions[ran[i]]==-1 else 1)/abs(np.sum(c.storage_TS[ran[i]][subset_reduced[i]])) for i in range(len(ran))])

        for i in range(len(ran)) : 
            c.storage_TS[ran[i]][subset_reduced[i]]=c.storage_TS[ran[i]][subset_reduced[i]]*(1+coeffs[i])
            c.storage_sum[ran[i]]=-np.sum(np.where(c.storage_TS[ran[i]]<0,c.storage_TS[ran[i]],0))
    
    if (total_Y_movable_load!=0 ) :        
        c.Y_DSM = c.Y_DSM*total_Y_movable_load/sum(c.Y_DSM)
             
    c.D_DSM[D_DSM_indexes,:] = np.multiply(c.D_DSM[D_DSM_indexes,:].T,total_D_Movable_load[D_DSM_indexes]/np.sum(c.D_DSM[D_DSM_indexes,:],axis=1)).T
        
    return (c)

def check_bouclage(c,storage_characteristics):
    bouclages=[]
    for i in range(len(c.storage_TS)):
        bouclages.append(sum(np.where(c.storage_TS[i]>0,c.storage_TS[i],0))+storage_characteristics[4,i]*sum(np.where(c.storage_TS[i]<0,c.storage_TS[i],0)))
    return(sum(bouclages))
#def bouclages(c,n_store,storage_characteristics,total_Y_movable_load,total_D_Movable_load,D_DSM_indexes ):

####Correction des TS pour boucler le stockage

#    ind_discharge=[np.where(c.storage_TS[store]>0)[0] for store in range(n_store)]
#    absolutes = np.sum(abs(c.storage_TS),axis=1)
#    lenghts = np.array([len(ind_discharge[store]) for store in range(n_store)])
#    test_sign = np.array([np.any(c.storage_TS[store]>0) &  np.any(c.storage_TS[store]<0) for store in range(n_store)])
    
   # ind_dis = np.where(c.storage_TS>0,1,0)
        
   # sum_dis=np.sum(ind_dis,axis=1)
   # sum_cha=len(c.storage_TS[0])-sum_dis
    
    #On met tout à 0 si tout de même signe   
  #  test_sign=np.maximum(((sum_cha==0)*1.), ( (sum_dis==0)*1.))
    
#    if not(test_sign.all()):
#        c.storage_TS=(c.storage_TS.T*(1.*test_sign)).T
#        ind_dis[:,0]=ind_dis[:,0]-2*test_sign*(ind_dis[:,0]-0.5)
 
#    storage_charge=-np.sum(np.where(c.storage_TS<0,c.storage_TS,0),axis=1) 
#    storage_discharge=np.sum(np.where(c.storage_TS>0,c.storage_TS,0),axis=1)    

#    ind_cha=1-ind_dis
    
#    volumes = np.sum(((ind_cha*c.storage_TS).T*storage_characteristics['Round-trip efficiency']).T+ind_dis*c.storage_TS,axis=1)


#    len_subset = max(1,min(40,min(lenghts),min(len(c.storage_TS[0])-lenghts)))
#    mat_choices = np.random.randint(0,min(lenghts),len_subset)
#    indexes = ind_discharge
    
#    subset = c.storage_TS[np.ix_(range(n_store),mat_choices)]
#    coeffs=[]
#    for store in range(n_store):
#        if ((c.storage_sum[store]>0) & (sum(c.storage_TS[store]>0)>0)) :
#            if volumes[store]>0:
#                coeffs.append(1+volumes[store]/sum(c.storage_TS[store,subset[store]]))
#            else : 
#                coeffs.append(1+volumes[store]/sum(c.storage_TS[store,subset[store]]))
#        else :
#            coeffs.append(1.)
#        c.storage_TS[store,subset[store]] = c.storage_TS[store,subset[store]]*coeffs[store]
        
#    t2=time.time()
    
#    c.storage_sum=-np.sum(np.where(c.storage_TS<0,c.storage_TS,0),axis=1)

#    t3=time.time()

 #   print(t3-t2,t2-t1,t1-t0)
    
  #  if (total_Y_movable_load!=0 ) :        
  #      c.Y_DSM = c.Y_DSM*total_Y_movable_load/sum(c.Y_DSM)
             
  #  c.D_DSM[D_DSM_indexes[0],:] = np.multiply(c.D_DSM[D_DSM_indexes[0],:].T,total_D_Movable_load[D_DSM_indexes[0]]/np.sum(c.D_DSM[D_DSM_indexes[0],:],axis=1)).T
        
  #  return (c)

def initial_population_avec_pertes_contraintes_3(inputs):
    (n_bits,n_pop,n_store,time_resolution,Bounds_prod,sum_load,Y_movable_load,D_movable_load,storage_characteristics,constraint_num, constraint_level,n_contracts)=tuple(inputs[i] for i in range(12))
    stored_volumes = np.float64(np.multiply(np.random.choice([0,1],(n_pop,n_store),p=[0.1,0.9]),abs(np.random.normal(sum_load/n_store*(constraint_level/1.5)**1.5,sum_load/n_store*(constraint_level/2)**2,(n_pop,n_store)))))
    Initial_prod = [[np.random.randint(0,Bound,1)[0] for Bound in Bounds_prod] for j in range(n_pop)]
    Initial_prod[0:min(n_pop,20)]=[((i+11)*Bounds_prod/30).astype(int) for i in range(min(n_pop,20))]
    Initial_contracts = np.random.randint(0,n_contracts,n_pop)
    
    Initial_YDSM = [np.random.rand(n_bits) for i in range(n_pop)]
    Initial_YDSM = [Initial_YDSM[i]/sum(Initial_YDSM[i])*sum(Y_movable_load) for i in range(n_pop)]
    Initial_DDSM = [[np.random.rand(int(time_resolution*24)) for j in range(int(n_bits/time_resolution/24))] for i in range(n_pop)]
    Initial_DDSM = [np.array([Initial_DDSM[i][j]/sum(Initial_DDSM[i][j])*np.sum(D_movable_load[j*int(time_resolution*24):((j+1)*int(time_resolution*24))]) for j in range(int(n_bits/time_resolution/24))]) for i in range(n_pop)]
    Initial_population = list()
    
    for j in range(n_pop):
        Initial_storage_power=np.zeros((n_store,n_bits))
        for i in range(n_store):
            where_charge=np.random.choice([-1,1],n_bits-1,p=[1/(1+storage_characteristics[4,:][i]),storage_characteristics[4,:][i]/(1+storage_characteristics[4,:][i])])
            where_charge=np.append(where_charge,np.array((-where_charge[np.random.choice(n_bits-1)])))
            Initial_storage_power[i,:] = where_charge*abs(np.random.normal(2,1,n_bits))
            Initial_storage_power[i,:][Initial_storage_power[i,:]>0] = np.round(Initial_storage_power[i,:][Initial_storage_power[i,:]>0]/sum(Initial_storage_power[i,:][Initial_storage_power[i,:]>0])*stored_volumes[j,i]*storage_characteristics[4,:][i],7)
            Initial_storage_power[i,:][Initial_storage_power[i,:]<0] = -Initial_storage_power[i,:][Initial_storage_power[i,:]<0]/sum(Initial_storage_power[i,:][Initial_storage_power[i,:]<0])*stored_volumes[j,i]
        Init_pop_j=ECl.Non_JIT_Individual(production_set=np.array(Initial_prod[j],dtype=np.int64),storage_sum=stored_volumes[j],storage_TS=Initial_storage_power,contract=Initial_contracts[j],Y_DSM=Initial_YDSM[j],D_DSM=Initial_DDSM[j],fitness=np.nan,trades=np.full([n_bits], np.float64(np.nan)))
        Initial_population.append(Init_pop_j)
       
    return(Initial_population)   

def combining_HD_solutions (D_solution,H_solution,D_time_resolution,H_time_resolution,n_days,Contexte):
    
    weights = np.random.random(len(D_solution.production_set)+1)
# perform crossover
    HD_production_set=(weights[0:(len(D_solution.production_set))]*D_solution.production_set+(1-weights[0:(1*len(D_solution.production_set))])*H_solution.production_set).round().astype(int)
    HD_production_set=np.maximum(D_solution.production_set,H_solution.production_set)

    HD_storage_TS_base = np.tile(H_solution.storage_TS,int(n_days))
    HD_storage_TS = HD_storage_TS_base+np.repeat(D_solution.storage_TS/H_time_resolution/24, int(H_time_resolution/D_time_resolution)).reshape(2,int(n_days*H_time_resolution*24))
    HD_contract = D_solution.contract if weights[len(D_solution.production_set)]<0.5 else H_solution.contract
    
    HD_Y_DSM = np.tile(H_solution.Y_DSM,int(n_days))+np.repeat(D_solution.Y_DSM/H_time_resolution/24, int(H_time_resolution/D_time_resolution))
    HD_D_DSM = np.tile(H_solution.D_DSM,int(n_days)).reshape(int(n_days),int(24*H_time_resolution))+np.repeat(D_solution.D_DSM/H_time_resolution/24, int(H_time_resolution/D_time_resolution)).reshape(int(n_days),int(24*H_time_resolution))
    HD_storage_sum = -np.sum(np.where(HD_storage_TS<0,HD_storage_TS,0),axis=1)
    
    coeffs = np.multiply(HD_storage_sum,np.divide(Contexte.storage_characteristics['Round-trip efficiency'],1*(HD_storage_sum==0)+np.sum(np.where(HD_storage_TS>0,HD_storage_TS,0),axis=1)))
    HD_storage_TS=np.where(HD_storage_TS>0,np.multiply(coeffs,HD_storage_TS.T).T,HD_storage_TS)
    
    HD_solution=ECl.Individual(production_set=HD_production_set,storage_sum=HD_storage_sum,storage_TS=HD_storage_TS,contract=HD_contract,Y_DSM=HD_Y_DSM,D_DSM=HD_D_DSM,fitness=np.NAN)

    return(HD_solution)

def combining_HD_solutions2 (D_solutions,H_solutions,days,D_time_resolution,H_time_resolution,n_days,Contexte):

    days_ordered = np.sort(days)
    H_solutions_ordered = [H_solutions[i] for i in np.argsort(days)] 
    n_hourly_solutions = len(H_solutions_ordered)
    
    n_prod = len(D_solutions[0].production_set)
    n_store = len(D_solutions[0].storage_TS)
    
    choices_contract = np.random.choice(range(5),len(D_solutions))
    weights = np.random.random(len(D_solutions)*n_prod)
    weights_prod = np.random.rand(len(D_solutions),n_prod,4)
    weights_prod2 = (weights_prod.T/(np.sum(weights_prod,axis=2).T)).T
# perform crossover
    HD_population =[]
    for i in range(len(D_solutions)):
    
        H_solution_seq = [H_solutions_ordered[j][i] for j in range(n_hourly_solutions)]
        D_solution = D_solutions[i]
    
        HD_production_set=(weights[(i*n_prod):((i+1)*n_prod)]*D_solution.production_set+(1-weights[(i*n_prod):((i+1)*n_prod)])*np.sum(weights_prod2[i].T*(np.array([H_solution_seq[k].production_set for k in range(4)])),axis=0)).round().astype(int)

        HD_storage_TS_begin = np.tile(H_solution_seq[0].storage_TS,int(days_ordered[0]))   
        t1=[]
        t1b=[]
        for j in range(n_hourly_solutions-1) :
            t1.append(np.linspace(H_solution_seq[j].storage_TS, H_solution_seq[j+1].storage_TS,1+days_ordered[j+1]-days_ordered[j])[:-1])
            t1b.append(np.hstack(t1[j]))
       # t2=np.linspace(H_solution_seq[1].storage_TS, H_solution_seq[2].storage_TS,1+days_ordered[2]-days_ordered[1])[:-1]
       # t2b=np.hstack(t2)
       # t3=np.linspace(H_solution_seq[2].storage_TS, H_solution_seq[3].storage_TS,1+days_ordered[3]-days_ordered[2])[:-1]
       # t3b=np.hstack(t3)
        HD_storage_TS_end = np.tile(H_solution_seq[n_hourly_solutions-1].storage_TS,int(n_days)-int(days_ordered[n_hourly_solutions-1]))    

        HD_storage_TS_base = np.hstack((HD_storage_TS_begin,np.hstack(t1b),HD_storage_TS_end))
    
        HD_storage_TS = HD_storage_TS_base+np.repeat(D_solution.storage_TS/H_time_resolution/24, int(H_time_resolution/D_time_resolution)).reshape(n_store,int(n_days*H_time_resolution*24))

        HD_contract = D_solution.contract if choices_contract[i]==4 else [H_solution_seq[k].contract for k in range(4) ][choices_contract[i]]
    
        YDSM_begin = np.tile(H_solution_seq[0].Y_DSM ,int(days_ordered[0])) 
        YDSM1=[]
        YDSM1b=[]
        for j in range(n_hourly_solutions-1) :
            YDSM1.append(np.linspace(H_solution_seq[j].Y_DSM, H_solution_seq[j+1].Y_DSM,1+days_ordered[j+1]-days_ordered[j])[:-1])
            YDSM1b.append(np.hstack(YDSM1[j]))
        #YDSM1=np.linspace(H_solution_seq[0].Y_DSM, H_solution_seq[1].Y_DSM,1+days_ordered[1]-days_ordered[0])[:-1]
        #YDSM1b=np.hstack(YDSM1)
        #YDSM2=np.linspace(H_solution_seq[1].Y_DSM, H_solution_seq[2].Y_DSM,1+days_ordered[2]-days_ordered[1])[:-1]
        #YDSM2b=np.hstack(YDSM2)
        #YDSM3=np.linspace(H_solution_seq[2].Y_DSM, H_solution_seq[3].Y_DSM,1+days_ordered[3]-days_ordered[2])[:-1]
        #YDSM3b=np.hstack(YDSM3)
        YDSM_end = np.tile(H_solution_seq[n_hourly_solutions-1].Y_DSM,int(n_days)-int(days_ordered[n_hourly_solutions-1]))    

        YDSM_base = np.hstack((YDSM_begin,np.hstack(YDSM1b),YDSM_end))

        YDSM = YDSM_base+np.repeat(D_solution.Y_DSM/H_time_resolution/24, int(H_time_resolution/D_time_resolution))

        DDSM_begin = np.tile(H_solution_seq[0].D_DSM ,int(days_ordered[0]))   
        DDSM1=[]
        DDSM1b=[]
        for j in range(n_hourly_solutions-1) :
            DDSM1.append(np.linspace(H_solution_seq[j].D_DSM, H_solution_seq[j+1].D_DSM,1+days_ordered[j+1]-days_ordered[j])[:-1])
            DDSM1b.append(np.hstack(DDSM1[j]))
        #DDSM2=np.linspace(H_solution_seq[1].D_DSM, H_solution_seq[2].D_DSM,1+days_ordered[2]-days_ordered[1])[:-1]
        #DDSM2b=np.hstack(DDSM2)
        #DDSM3=np.linspace(H_solution_seq[2].D_DSM, H_solution_seq[3].D_DSM,1+days_ordered[3]-days_ordered[2])[:-1]
        #DDSM3b=np.hstack(DDSM3)
        DDSM_end = np.tile(H_solution_seq[n_hourly_solutions-1].D_DSM,int(n_days)-int(days_ordered[n_hourly_solutions-1]))    
        
        DDSM_base = np.hstack((DDSM_begin,np.hstack(DDSM1b),DDSM_end))

        DDSM = (DDSM_base+np.repeat(D_solution.D_DSM/H_time_resolution/24, int(H_time_resolution/D_time_resolution))).reshape(int(n_days),int(24*H_time_resolution))
          
        HD_storage_sum = -np.sum(np.where(HD_storage_TS<0,HD_storage_TS,0),axis=1)
    
        if (np.any(HD_storage_TS>0)) :
            coeffs = np.multiply(HD_storage_sum,np.divide(Contexte.storage_characteristics[4,:],1*(HD_storage_sum==0)+np.sum(np.where(HD_storage_TS>0,HD_storage_TS,0),axis=1)))
            HD_storage_TS=np.where(HD_storage_TS>0,np.multiply(coeffs,HD_storage_TS.T).T,HD_storage_TS)
    
        HD_population.append(ECl.Non_JIT_Individual(production_set=HD_production_set,storage_sum=HD_storage_sum,storage_TS=HD_storage_TS,contract=HD_contract,Y_DSM=YDSM,D_DSM=DDSM,fitness=np.float64(np.nan),trades=np.empty(np.int64(n_days*H_time_resolution*24))* np.float64(np.nan)))

    return(HD_population)


