# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 19:23:34 2023

@author: jlegalla
"""
import ERMESS_classes as ECl
import ERMESS_operators as Eop
import numpy as np
from numba import jit
import PMS
    
        
def pro_to_research(pop_pro,Contexte):
    pop_res = []
    for ind_pro in pop_pro:
        production = ((Contexte.prods_U.T*ind_pro.production_set).sum(axis=1)+Contexte.prod_C)/1000    
        (storage_TS,trades,D_DSM,Y_DSM,SOCs_eff,losses,P_diff) = PMS.LFE_CCE.py_func(ind_pro, Contexte.Non_movable_load, Contexte.total_D_Movable_load, Contexte.total_Y_Movable_load, production , Contexte.n_bits,Contexte.n_store,Contexte.time_resolution, Contexte.Connexion, Contexte.storage_characteristics)
        D_DSM = D_DSM.reshape((int(Contexte.n_bits/(Contexte.time_resolution*24)),int(Contexte.time_resolution*24)))
        storage_sum=np.array([-np.sum(np.where(storage_TS[i]<0,storage_TS[i],0)) for i in range(Contexte.n_store)])
        ind_res = ECl.Non_JIT_Individual_res(ind_pro.production_set, storage_sum, storage_TS, ind_pro.contract, Y_DSM, D_DSM, np.nan, trades)
        pop_res.append(ind_res)
    return (pop_res)
    
# tournament selection
def selection2(pop, k=3):
 # first random selection
    selection_ix = np.random.randint(len(pop))
    for ix in np.random.randint(0, len(pop), k-1):
 # check if better (e.g. perform a tournament)
        if pop[ix].fitness < pop[selection_ix].fitness:
            selection_ix = ix
    return pop[selection_ix]


@jit(nopython=True)
def crossover_reduit(p1, p2, r_cross,n_bits,groups, n_store,storage_characteristics):
 
    # children are copies of parents by default
    c1 = ECl.Individual_res(p1.production_set,p1.storage_sum,p1.storage_TS,p1.contract,p1.Y_DSM,p1.D_DSM,np.float64(np.nan),np.full(n_bits,np.nan)) 
    c2 = ECl.Individual_res(p2.production_set,p2.storage_sum,p2.storage_TS,p2.contract,p2.Y_DSM,p2.D_DSM,np.float64(np.nan),np.full(n_bits,np.nan)) 

    cross_rand = np.random.rand()
 # check for recombination
    if cross_rand < r_cross:
    # select random weights

        weights = np.random.random(5+len(groups))
 # perform crossover
 
        mask_prod = weights[5:len(weights)]<0.5
        cross_indexes = np.full(len(c1.production_set),-1,dtype=np.int64)
        for k in range(len(mask_prod)):
            if mask_prod[k]:
                int_idx=groups[k]
                cross_indexes[int_idx] =int_idx

        c1.production_set=np.where(cross_indexes>=0,p2.production_set,p1.production_set)
        c2.production_set=np.where(cross_indexes>=0,p1.production_set,p2.production_set)
        
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
    
def NON_JIT_mutation_contraintes_3(c, random_factors, choices,n_bits, Bounds_prod, groups, Non_movable_load, constraint_num, constraint_level, prods_U,prod_C, n_store, n_contracts, time_resolution, storage_characteristics,Volums_prod,D_DSM_indexes,activate_Y_DSM,hyperparameters_operators_num):
       
        usage_ope = np.repeat(0, 29)
        #MUTATION DU CONTRAT     
        if ((random_factors[0]<hyperparameters_operators_num[0,0])) :  
              c=Eop.contract_operator(c,n_contracts)
              usage_ope[0]=1
              
        ## Semi-driven operator
        ## DIMINUTION DE LA PUISSANCE DU CONTRAT
        if ((random_factors[1]<hyperparameters_operators_num[0,0]) & np.any(c.trades>0)) :
            c=Eop.power_contract_operator(c,choices)
            usage_ope[1]=1

              
        if (random_factors[2]<hyperparameters_operators_num[0,1]) :            
             c=Eop.production_operator(c,choices,Bounds_prod,groups,prods_U,n_bits,hyperparameters_operators_num)    
             usage_ope[2]=1
             
             
        if (random_factors[3]<hyperparameters_operators_num[0,1]/2) :
             c=Eop.production_switch_operator(c,Bounds_prod,groups)
             usage_ope[3]=1
                  
        if (random_factors[4]<hyperparameters_operators_num[0,1]/10) :
                  c=Eop.production_ingroup_switch_operator(c,Bounds_prod,groups)
                  usage_ope[4]=1
        
        #MUTATIONS DES SERIES TEMPORELLES
        #RANDOM OPERATORS
        if ((random_factors[5]<hyperparameters_operators_num[0,6])) :
                  c=Eop.timeserie_operator(c,choices,hyperparameters_operators_num)
                  usage_ope[5]=1
        
            
        # Mutation aléatoire de la série temporelle sur des séquences voisines en respectant les sens
        if ((random_factors[6]<hyperparameters_operators_num[0,6])) :
                  c=Eop.timeserie_sequences_operator(c,choices,hyperparameters_operators_num)
                  usage_ope[6]=1

        if ((random_factors[7]<hyperparameters_operators_num[0,6]) & (time_resolution>(1/10))) :                  
                  c = Eop.timeserie_daypattern_operator(c,choices,time_resolution,n_bits,hyperparameters_operators_num)
                  usage_ope[7]=1
                              
        ##MODIFICATION DE L'UTILISATION GLOBALE DES STOCKAGES
        if ((random_factors[8]<hyperparameters_operators_num[0,3])) :
              c=Eop.storage_use_global_operator(c,choices,n_store,hyperparameters_operators_num)
              usage_ope[8]=1
              
              
        ##MODIFICATION DE L'UTILISATION DES STOCKAGES SUR DES SOUS-ENSEMBLES
        if (random_factors[9]<(hyperparameters_operators_num[0,3])):
               c=Eop.storage_use_local_operator(c,choices,hyperparameters_operators_num)
               usage_ope[9]=1
               
            
               #On introduit un transfert éventuel entre stockages
        if ((n_store>1) & (random_factors[10]<hyperparameters_operators_num[0,8]) ) :
                         c=Eop.storage_transfer_1_operator(c,n_store,random_factors,hyperparameters_operators_num)
                         usage_ope[10]=1
                                                  
                #On introduit un transfert éventuel entre stockages_v2  
        if ((n_store>1) & (random_factors[11]<hyperparameters_operators_num[0,8])) :
                 c=Eop.storage_transfer_2_operator(c,n_store,storage_characteristics,hyperparameters_operators_num)
                 usage_ope[11]=1        
        
        ##SEMI-ORIENTED OPERATORS
        ## DIMINUTION DU VOLUME D'UN STOCKAGE ALEATOIRE  
        if ((random_factors[12]<hyperparameters_operators_num[0,2])& (sum(c.storage_TS[choices[6]])!=0)) :
            c=Eop.storage_volume_operator(c,choices,storage_characteristics,hyperparameters_operators_num)
            usage_ope[12]=1                


        ## DIMINUTION DE LA PUISSANCE D'UN STOCKAGE ALEATOIRE
        if (random_factors[13]<(hyperparameters_operators_num[0,4])) :
            c=Eop.storage_power_operator(c,choices,hyperparameters_operators_num)        
            usage_ope[13]=1


        ##ANNULATION DES MOUVEMENTS OPPOSES DE 2 STOCKAGES ALEATOIRES
        if (n_store>1) & (random_factors[14]<hyperparameters_operators_num[0,7]):
              c=Eop.opposite_moves_operator(c,n_store,hyperparameters_operators_num)
              usage_ope[14]=1
         
    ### ANNULATION DES DECHARGES/EXPORT ou CHARGES/IMPORTS 
        if (random_factors[15]<hyperparameters_operators_num[0,5]):
             c=Eop.Scheduling_consistency_operator(c,choices,hyperparameters_operators_num)
             usage_ope[15]=1   


        ## Long-term consistency
        if ((random_factors[16]<hyperparameters_operators_num[0,12]) & (len(c.storage_TS[0])>(100*int(time_resolution*24)))):
            c=Eop.Long_term_consistency_operator(c,choices,random_factors,time_resolution,hyperparameters_operators_num)
            usage_ope[16]=1
        

            ###Interdaily consistency
        if ((random_factors[17]<hyperparameters_operators_num[0,12]) & (len(c.storage_TS[0])>(20*int(time_resolution*24)))):
            c=Eop.Interdaily_consistency_operator(c,choices,random_factors,n_bits,time_resolution)
            usage_ope[17]=1
        
        ##APPLATISSEMENT DES COURBES
        if (random_factors[18]<hyperparameters_operators_num[0,10]):
               c=Eop.curve_smoothing_operator(c,choices,random_factors,hyperparameters_operators_num)
               usage_ope[18]=1
               
        #Specification des rôles des stockages
        if  ((random_factors[19]<hyperparameters_operators_num[0,9]) & n_store>1):
            c=Eop.storage_specification_operator(c,random_factors,n_store,storage_characteristics,hyperparameters_operators_num)
            usage_ope[19]=1
   
            ## OPERATEUR DE CONTRAINTE
        if (random_factors[20]<hyperparameters_operators_num[0,11]) :
            c=Eop.constraint_operator(c,constraint_num,hyperparameters_operators_num)
            usage_ope[20]=1
                
        #Mutation du DSM 
        if ((random_factors[21]<hyperparameters_operators_num[0,13]) & activate_Y_DSM) :
              c=Eop.Y_DSM_operator(c,hyperparameters_operators_num)
              usage_ope[21]=1
        if ((random_factors[22]<hyperparameters_operators_num[0,10]) & activate_Y_DSM) :
              c=Eop.Y_DSM_smoothing_operator(c,random_factors,hyperparameters_operators_num)
              usage_ope[22]=1
        if ((random_factors[23]<hyperparameters_operators_num[0,12]) & (len(c.Y_DSM)>(20*int(time_resolution*24))) & activate_Y_DSM):
              c=Eop.Y_DSM_Interdaily_consistency_operator(c,random_factors,n_bits,time_resolution)
              usage_ope[23]=1   
        if ((random_factors[24]<hyperparameters_operators_num[0,13]) & activate_Y_DSM):
              c=Eop.Y_DSM_oriented_operator(c,choices,random_factors,hyperparameters_operators_num)
              usage_ope[24]=1   
        if ((random_factors[25]<hyperparameters_operators_num[0,14]) & (len(D_DSM_indexes)>0) & (c.D_DSM.shape[1]>1)) :
                c=Eop.D_DSM_operator(c,D_DSM_indexes,hyperparameters_operators_num)
                usage_ope[25]=1 
        if ((random_factors[26]<hyperparameters_operators_num[0,10]) & (len(D_DSM_indexes)>0) & (c.D_DSM.shape[1]>1)) :
              c=Eop.D_DSM_smoothing_operator(c,D_DSM_indexes,random_factors,hyperparameters_operators_num)
              usage_ope[26]=1
        if ((random_factors[27]<hyperparameters_operators_num[0,12]) & (len(D_DSM_indexes)>50) & (c.D_DSM.shape[1]>1)):
              c=Eop.D_DSM_Interdaily_consistency_operator(c,D_DSM_indexes,random_factors,time_resolution)
              usage_ope[27]=1       
        if ((random_factors[28]<hyperparameters_operators_num[0,14]) & (len(D_DSM_indexes)>0) & (c.D_DSM.shape[1]>1)) :
              c=Eop.D_DSM_oriented_operator(c,D_DSM_indexes,time_resolution,choices,random_factors,hyperparameters_operators_num)
              usage_ope[28]=1
         
        return(c,usage_ope)
#        return(c)

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
def bouclages(c,n_store,storage_characteristics,total_Y_movable_load,total_D_Movable_load,D_DSM_indexes,activate_Y_DSM ):

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
    
    if (activate_Y_DSM):    
        c.Y_DSM = c.Y_DSM*total_Y_movable_load/sum(c.Y_DSM)
             
    c.D_DSM[D_DSM_indexes,:] = np.multiply(c.D_DSM[D_DSM_indexes,:].T,total_D_Movable_load[D_DSM_indexes]/np.sum(c.D_DSM[D_DSM_indexes,:],axis=1)).T
        
    return (c)

def check_bouclage(c,storage_characteristics):
    bouclages=[]
    for i in range(len(c.storage_TS)):
        bouclages.append(sum(np.where(c.storage_TS[i]>0,c.storage_TS[i],0))+storage_characteristics[4,i]*sum(np.where(c.storage_TS[i]<0,c.storage_TS[i],0)))
    return(sum(bouclages))

def initial_population_avec_pertes_contraintes_3(inputs):
    (n_bits,n_pop,n_store,time_resolution,Bounds_prod,groups,sum_load,Y_movable_load,D_movable_load,storage_characteristics,constraint_num, constraint_level,n_contracts)=tuple(inputs[i] for i in range(13))
    stored_volumes = np.float64(np.multiply(np.random.choice([0,1],(n_pop,n_store),p=[0.1,0.9]),abs(np.random.normal(sum_load/n_store*(constraint_level/1.5)**1.5,sum_load/n_store*(constraint_level/2)**2,(n_pop,n_store)))))
    Initial_prod_index = np.random.rand(n_pop,len(Bounds_prod))        
    Initial_prod = np.array([[np.random.randint(0,Bound,1)[0] for Bound in Bounds_prod] for j in range(n_pop)])
    Initial_prod[0:min(n_pop,20)]=[((i+11)*Bounds_prod/30).astype(int) for i in range(min(n_pop,20))]
    Initial_contracts = np.random.randint(0,n_contracts,n_pop)
    
    Initial_YDSM = [np.random.rand(n_bits) for i in range(n_pop)]
    Initial_YDSM = [Initial_YDSM[i]/sum(Initial_YDSM[i])*sum(Y_movable_load) for i in range(n_pop)]
    Initial_DDSM = [[np.random.rand(int(time_resolution*24)) for j in range(int(n_bits/time_resolution/24))] for i in range(n_pop)]
    Initial_DDSM = [np.array([Initial_DDSM[i][j]/sum(Initial_DDSM[i][j])*np.sum(D_movable_load[j*int(time_resolution*24):((j+1)*int(time_resolution*24))]) for j in range(int(n_bits/time_resolution/24))]) for i in range(n_pop)]
    Initial_population = list()
    
    for j in range(n_pop):
        ones_prod=[groups[i][np.argmax(Initial_prod_index[j][groups[i]])] for i in range(len(groups))]
        Initial_prod[j][np.array([i not in ones_prod for i in range(len(Bounds_prod))])]=0      
        Initial_storage_power=np.zeros((n_store,n_bits))
        for i in range(n_store):
            where_charge=np.random.choice([-1,1],n_bits-1,p=[1/(1+storage_characteristics[4,:][i]),storage_characteristics[4,:][i]/(1+storage_characteristics[4,:][i])])
            where_charge=np.append(where_charge,np.array((-where_charge[np.random.choice(n_bits-1)])))
            Initial_storage_power[i,:] = where_charge*abs(np.random.normal(2,1,n_bits))
            Initial_storage_power[i,:][Initial_storage_power[i,:]>0] = np.round(Initial_storage_power[i,:][Initial_storage_power[i,:]>0]/sum(Initial_storage_power[i,:][Initial_storage_power[i,:]>0])*stored_volumes[j,i]*storage_characteristics[4,:][i],7)
            Initial_storage_power[i,:][Initial_storage_power[i,:]<0] = -Initial_storage_power[i,:][Initial_storage_power[i,:]<0]/sum(Initial_storage_power[i,:][Initial_storage_power[i,:]<0])*stored_volumes[j,i]
        Init_pop_j=ECl.Non_JIT_Individual_res(production_set=Initial_prod[j],storage_sum=stored_volumes[j],storage_TS=Initial_storage_power,contract=Initial_contracts[j],Y_DSM=Initial_YDSM[j],D_DSM=Initial_DDSM[j],fitness=np.nan,trades=np.full([n_bits], np.float64(np.nan)))
        Initial_population.append(Init_pop_j)
       
    return(Initial_population)   

def combining_HD_solutions2 (D_solutions,H_solutions,days,D_time_resolution,H_time_resolution,n_days,Contexte):

    days_ordered = np.sort(days)
    H_solutions_ordered = [H_solutions[i] for i in np.argsort(days)] 
    n_hourly_solutions = len(H_solutions_ordered)
    
    n_prod = len(D_solutions[0].production_set)
    n_store = len(D_solutions[0].storage_TS)
    
    choices_contract = np.random.choice(range(5),len(D_solutions))
 #   weights = np.random.random(len(Contexte.groups),(len(D_solutions)))
 #   mask_prod = np.array([sum(weights>0.5,weights>0.625,weights>0.75,weights>0.875)]).transpose(2,0,1)
    indexes_solution_prod = np.random.choice (range(5),(len(D_solutions),len(Contexte.groups)),replace=True,p=(0.5,0.125,0.125,0.125,0.125))

    weights_prod = np.random.rand(len(D_solutions),n_prod,4)
    weights_prod2 = (weights_prod.T/(np.sum(weights_prod,axis=2).T)).T
# perform crossover
    HD_population =[]
    for i in range(len(D_solutions)):
    
        H_solution_seq = [H_solutions_ordered[j][i] for j in range(n_hourly_solutions)]
        D_solution = D_solutions[i]
    
   ##     HD_production_set=(weights[(i*n_prod):((i+1)*n_prod)]*D_solution.production_set+(1-weights[(i*n_prod):((i+1)*n_prod)])*np.sum(weights_prod2[i].T*(np.array([H_solution_seq[k].production_set for k in range(4)])),axis=0)).round().astype(int)
        
        HD_production_set=np.zeros(n_prod)
        for k in range(len(Contexte.groups)) :
            HD_production_set[Contexte.groups[k]]=[D_solutions[i].production_set,H_solution_seq[0].production_set,H_solution_seq[1].production_set,H_solution_seq[2].production_set,H_solution_seq[3].production_set][indexes_solution_prod[i][k]]      

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
    
        HD_population.append(ECl.Non_JIT_Individual_res(production_set=HD_production_set,storage_sum=HD_storage_sum,storage_TS=HD_storage_TS,contract=HD_contract,Y_DSM=YDSM,D_DSM=DDSM,fitness=np.float64(np.nan),trades=np.empty(np.int64(n_days*H_time_resolution*24))* np.float64(np.nan)))

    return(HD_population)


