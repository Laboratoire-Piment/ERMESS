# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:17:27 2025

@author: jlegalla
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def contract_operator(c,n_contracts):
    c.contract=np.random.randint(0,n_contracts,1)[0]
    return(c)

@jit(nopython=True)
def power_contract_operator(c,choices):
    store = choices[0]
    arg = np.argmax(c.trades)      
    c.storage_TS[store][arg] = c.storage_TS[store][arg]+c.trades[arg]*np.random.uniform(0,0.1)        
      
    return(c)

@jit(nopython=True)
def production_operator(c,choices,Bounds_prod,groups,prods_U,n_bits,hyperparameters_operators_num):    
    group = np.random.randint(len(groups))
    indexes = groups[group]
    productor = indexes[c.production_set[indexes]>0]
    productor = productor[0] if len(productor)>0 else np.random.choice(indexes)
    
    modifier=max(0,min(Bounds_prod[productor],c.production_set[productor]+np.random.randint(min(-2,-int(Bounds_prod[productor]/hyperparameters_operators_num[1,1])),max(3,int(Bounds_prod[productor]/hyperparameters_operators_num[1,1])))))
    added_prod = (modifier-c.production_set[productor])*prods_U[productor]*np.random.uniform(0,1,1)*np.random.normal(1,0.2,n_bits)/1000    
    c.production_set[productor]=modifier         
    places = np.where(c.trades>0)[0]
    store = choices[1]
    c.storage_TS[store,places] = c.storage_TS[store,places]-added_prod[places]
    return(c)

@jit(nopython=True)
def production_switch_operator(c,Bounds_prod,groups):
    positive_productors = np.where(c.production_set>0)[0]
    if len(positive_productors)>1:
        productors = np.random.choice(positive_productors,2,replace=False)
        c.production_set[productors[0]]=max(min(c.production_set[productors[0]]-1,Bounds_prod[productors[0]]),0)
        c.production_set[productors[1]]=max(min(c.production_set[productors[1]]+1,Bounds_prod[productors[1]]),0)  
    return(c)

@jit(nopython=True)
def production_ingroup_switch_operator(c,Bounds_prod,groups):
    group = np.random.randint(len(groups))
    indexes = groups[group]
    if ((len(indexes)>1) & max(c.production_set[indexes])>0):
        mask_NULL = c.production_set[indexes]==0
        productor_new = np.random.choice(indexes[mask_NULL],1)[0]
        c.production_set[indexes]=np.repeat(0,len(indexes))
        c.production_set[productor_new]=np.random.randint(0,Bounds_prod[productor_new]) 
    return(c)
                
@jit(nopython=True)
def timeserie_operator(c,choices,hyperparameters_operators_num):
    store = choices[2] 
    mutations=np.random.choice(np.arange(len(c.storage_TS[store])), size=np.random.randint(1,int(len(c.storage_TS[store])/hyperparameters_operators_num[1,6])+4), replace=False)
    c.storage_TS[store][mutations]=(c.storage_TS[store][mutations]+np.random.normal(0,50*hyperparameters_operators_num[5,6],len(mutations)))*np.random.normal(1,hyperparameters_operators_num[5,6],len(mutations))
    return(c)

@jit(nopython=True)
def timeserie_sequences_operator(c,choices,hyperparameters_operators_num):
    store = choices[3]     
    len_subset = np.random.randint(1,min(30,max(3,int(len(c.storage_TS[0])/hyperparameters_operators_num[1,6]))))                  
    starts = np.random.choice(np.arange(0,len(c.storage_TS[0])-len_subset),np.random.randint(1,min(30,int(len(c.storage_TS[0])/(2*hyperparameters_operators_num[1,6]))+2)))                      
    places=np.empty((len(starts),len_subset),dtype=np.int64)
    for i in range(len(starts)):
        places[i]=np.arange(starts[i],starts[i]+len_subset)
    matrice_mult = np.random.normal(1,hyperparameters_operators_num[5,6],places.size).reshape(places.shape[0],places.shape[1])
    matrice_add = np.random.normal(0,hyperparameters_operators_num[5,6],places.size).reshape(places.shape[0],places.shape[1])
    for i in range(len(starts)):
        c.storage_TS[store,places[i]]=c.storage_TS[store,places[i]]*matrice_mult[i]+matrice_add[i]
    return(c)

@jit(nopython=True)
def timeserie_daypattern_operator(c,choices,time_resolution,n_bits,hyperparameters_operators_num):
    store = choices[4] 
    days = np.arange(int(n_bits/time_resolution/24))                      
    hours = np.random.choice(np.arange(0,int(time_resolution*24)),np.random.randint(1,int(time_resolution*24)),replace=False)

    places=np.empty((len(days),len(hours)),dtype=np.int64)
    for i in range(len(days)):
        places[i]=hours+days[i]*int(time_resolution*24)
    matrice_mult = np.random.normal(1,hyperparameters_operators_num[5,6],len(hours))
    matrice_add = np.random.normal(0,5*hyperparameters_operators_num[5,6],len(hours))
    for i in range(len(days)):
        c.storage_TS[store,places[i]]=c.storage_TS[store,places[i]]*matrice_mult+matrice_add
    return(c)

@jit(nopython=True)
def storage_use_global_operator(c,choices,n_store,hyperparameters_operators_num):
    store = choices[5]
    for store in range(n_store):
        c.storage_TS[store,] = c.storage_TS[store,]*np.random.uniform(1-1/hyperparameters_operators_num[6,3],1+1/hyperparameters_operators_num[6,3])
    return(c)

@jit(nopython=True)
def storage_use_local_operator(c,choices,hyperparameters_operators_num):
    store = choices[6]
    width = np.random.randint(1,40)
    len_subset = np.random.randint(1,max(3,int(len(c.storage_TS[0])/hyperparameters_operators_num[1,3])))              
    starts = np.random.choice(np.arange(0,len(c.storage_TS[0])-len_subset),width)
    places=np.empty((len(starts),len_subset),dtype=np.int64)
    for i in range(len(starts)):
        places[i]=np.arange(starts[i],starts[i]+len_subset)
    coeffs_modifier = (np.random.uniform(1-1/hyperparameters_operators_num[6,3],1+1/hyperparameters_operators_num[6,3],len(starts)),np.random.uniform(-100/hyperparameters_operators_num[6,3],100/hyperparameters_operators_num[6,3],len(starts)))
    for i in range(len(starts)):
        c.storage_TS[store,places[i]]=c.storage_TS[store,places[i]]*coeffs_modifier[0][i]+coeffs_modifier[1][i]
    return(c)

@jit(nopython=True)
def storage_transfer_1_operator(c,n_store,random_factors,hyperparameters_operators_num):
    (store_in,store_out)=np.random.choice(n_store,2,replace=False)
    if (sum(c.storage_sum[np.array([store_in,store_out])])>0):
        
             switch_indexes1=np.random.choice(len(c.storage_TS[0]) ,np.random.randint(2,max(3,int(len(c.storage_TS[0])/(hyperparameters_operators_num[1,8])))),replace=False)
             c.storage_TS[store_in,switch_indexes1]=np.sum(c.storage_TS[:,switch_indexes1],axis=0)*random_factors[40]*c.storage_sum[store_in]/sum(c.storage_sum[np.array([store_in,store_out])])
             c.storage_TS[store_out,switch_indexes1]=np.sum(c.storage_TS[:,switch_indexes1],axis=0)*(1-random_factors[40]*c.storage_sum[store_in]/sum(c.storage_sum[np.array([store_in,store_out])]))
    return(c)

@jit(nopython=True)
def storage_transfer_2_operator(c,n_store,storage_characteristics,hyperparameters_operators_num):
    (store_in,store_out)=np.random.choice(n_store,2,replace=False)
    len_subset = np.random.randint(1,min(30,max(3,int(len(c.storage_TS[0])/hyperparameters_operators_num[1,8]))))                  
    starts = np.random.choice(np.arange(0,len(c.storage_TS[0])-len_subset),np.random.randint(1,min(30,int(len(c.storage_TS[0])/(hyperparameters_operators_num[1,8]/2))+2)))              
    places=np.empty((len(starts),len_subset),dtype=np.int64)
    subset=np.empty((len(starts),len_subset),dtype=np.float64)
    for i in range(len(starts)):
        places[i]=np.arange(starts[i],starts[i]+len_subset)
        subset[i] = c.storage_TS[store_out,places[i]]
            
    direction = -1
    subset_cha = np.where(np.sign(subset)==direction,subset,0)
    switch_cha = -np.abs(np.random.normal(hyperparameters_operators_num[3,8],hyperparameters_operators_num[5,8],places.size).reshape(len(starts),len_subset))*subset_cha
    for i in range(len(starts)):
        c.storage_TS[store_in,places[i]] = c.storage_TS[store_in,places[i]]-switch_cha[i]
        c.storage_TS[store_out,places[i]] = c.storage_TS[store_out,places[i]]+switch_cha[i]

    direction = 1
    subset_dis = np.where(np.sign(subset)==direction,subset,0)
    switch_dis = np.abs(np.random.normal(hyperparameters_operators_num[3,8],hyperparameters_operators_num[5,8],places.size).reshape(len(starts),len_subset))*subset_dis
    for i in range(len(starts)):
        c.storage_TS[store_in,places[i]] = c.storage_TS[store_in,places[i]]+switch_dis[i]*storage_characteristics[4,store_in]/storage_characteristics[4,store_out]
        c.storage_TS[store_out,places[i]] = c.storage_TS[store_out,places[i]]-switch_dis[i]
    return(c)

@jit(nopython=True)
def storage_volume_operator(c,choices,storage_characteristics,hyperparameters_operators_num):
    store = choices[7]
    losses = np.where(c.storage_TS[store]/storage_characteristics[4,store]-c.storage_TS[store]>0,c.storage_TS[store]/storage_characteristics[4,store]-c.storage_TS[store],0)
    sum_diff_storages = np.cumsum(c.storage_TS[store]+losses) 
    points = np.argmin(sum_diff_storages),np.argmax(sum_diff_storages)
    if (points[0]!=points[1]):
        sub_storage = [np.concatenate((np.arange(min(points)),np.arange(max(points),len(c.storage_TS[0])))),np.arange(min(points),max(points))]
        changes = [np.random.choice(sub_storage[0],np.random.randint(max(2,int(len(sub_storage[0])/hyperparameters_operators_num[1,2]))),replace=False),np.random.choice(sub_storage[1],np.random.randint(1,max(2,int(len(sub_storage[1])/hyperparameters_operators_num[1,2]))),replace=False)]
        shift = np.random.uniform(0,np.ptp(sum_diff_storages)/hyperparameters_operators_num[6,2])

        if points[0]==min(points):
            c.storage_TS[store][changes[0]]=c.storage_TS[store][changes[0]]+shift*np.random.normal(hyperparameters_operators_num[3,2],hyperparameters_operators_num[5,2],len(changes[0]))/len(changes[0])
            c.storage_TS[store][changes[1]]=c.storage_TS[store][changes[1]]-shift*np.random.normal(hyperparameters_operators_num[3,2],hyperparameters_operators_num[5,2],len(changes[1]))/len(changes[1])
        else :
            c.storage_TS[store][changes[0]]=c.storage_TS[store][changes[0]]-shift*np.random.normal(hyperparameters_operators_num[3,2],hyperparameters_operators_num[5,2],len(changes[0]))/len(changes[0])
            c.storage_TS[store][changes[1]]=c.storage_TS[store][changes[1]]+shift*np.random.normal(hyperparameters_operators_num[3,2],hyperparameters_operators_num[5,2],len(changes[1]))/len(changes[1])
    return(c)

@jit(nopython=True)
def storage_power_operator(c,choices,hyperparameters_operators_num):
    store = choices[8]
    coeffs=np.random.uniform(1-1/hyperparameters_operators_num[6,4],1,2)
    for step in range(np.random.randint(max(1,int(len(c.storage_TS[0])/hyperparameters_operators_num[1,4])))):
        argmax = np.argmax(c.storage_TS[store])
        argmin = np.argmin(c.storage_TS[store])
        c.storage_TS[store][argmax] = c.storage_TS[store][argmax]*coeffs[0]
        c.storage_TS[store][argmin] = c.storage_TS[store][argmin]*coeffs[1]
    return(c)

@jit(nopython=True)
def opposite_moves_operator(c,n_store,hyperparameters_operators_num):
    stores = np.random.choice (n_store,2,replace=False)
    poss_changes = np.where(c.storage_TS[stores[0]]*c.storage_TS[stores[1]]<0)[0]
    if (len(poss_changes)>1):
        changes = np.random.choice(poss_changes, max(2,int(len(poss_changes)/hyperparameters_operators_num[1,7])),replace=False)
        c.storage_TS[stores[0],changes]=c.storage_TS[:,changes].sum(axis=0)
        c.storage_TS[stores[1],changes]=0
    return(c)

@jit(nopython=True)
def Scheduling_consistency_operator(c,choices,hyperparameters_operators_num):
    store = choices[9]
    len_subset = np.random.randint(1,min(100,max(3,int(len(c.storage_TS[0])/hyperparameters_operators_num[1,5]))))
    place = np.random.choice(np.arange(0,len(c.storage_TS[0])-len_subset))
    subset = c.storage_TS[store,np.arange(place,place+len_subset+1)]
    bornes = c.trades[np.arange(place,place+len_subset+1)]
    energy_residuals = np.where(subset*bornes<0,np.sign(bornes)*np.minimum(np.abs(subset),np.abs(bornes)),0)
    if ((energy_residuals!=0).any() & (energy_residuals==0).any()):
        c.storage_TS[store,np.arange(place,place+len_subset+1)]=np.where(subset*bornes<0,subset+energy_residuals,(subset-sum(energy_residuals)/sum(energy_residuals==0)))
    return(c)

#Apply the pattern of a random reference day to several days based on their proximity
@jit(nopython=True)
def Interdaily_consistency_operator(c,choices,random_factors,n_bits,time_resolution):
    n_days = int(n_bits/(24*time_resolution))
    store=choices[10]
    len_subset = int(time_resolution*24)
    starts_ref = np.random.choice(np.arange(n_days-1)),0,np.random.randint(1,n_days),np.random.choice(np.array([-1,1]))
    starts = np.arange(starts_ref[0],starts_ref[0]+starts_ref[2]+1)[::(-1)*starts_ref[3]]%n_days            
    places=np.empty((len(starts),len_subset),dtype=np.int64)
    for i in range(len(starts)):
        places[i]=np.arange(starts[i]*int(time_resolution*24)+starts_ref[1],starts[i]*int(time_resolution*24)+starts_ref[1]+(len_subset))    
    distances = np.abs(starts-starts[0])
    max_distances = max(distances)
    for i in range(len(starts)):
        c.storage_TS[store,places[i]] = ((1-random_factors[41]*(1-distances[i]/(2*max_distances)))*c.storage_TS[store,places[i]])+(random_factors[41]*(1-distances[i]/max_distances/2)*c.storage_TS[store,places[0]])
    return(c)

@jit(nopython=True)
def curve_smoothing_operator(c,choices,random_factors,hyperparameters_operators_num):
    store = choices[11]
    window_width = np.int64(min(6,len(c.storage_TS[0])/8))*2
    len_subset = np.random.randint(1,min(30,max(3,int(len(c.storage_TS[0])/hyperparameters_operators_num[1,10]))))              
    starts = np.random.choice(np.arange(window_width,len(c.storage_TS[0])-len_subset-window_width),np.random.randint(1,min(30,int(len(c.storage_TS[0])/hyperparameters_operators_num[6,10])+2)))              
    places=np.empty((len(starts),len_subset),dtype=np.int64)
    places_mva=np.empty((len(starts),len_subset+window_width),dtype=np.int64)
    cumsum_mat=np.empty((len(starts),len_subset+window_width),dtype=np.float64)
    for i in range(len(starts)):
        places[i]=np.arange(starts[i],starts[i]+len_subset)
        places_mva[i] = np.arange(starts[i]-window_width/2,starts[i]+len_subset+window_width/2)
        cumsum_mat[i] = np.cumsum(c.storage_TS[store,places_mva[i]]) 
    vec_places = places.flatten()    

    ma_vec = (cumsum_mat[:,window_width:] - cumsum_mat[:,:-window_width]) / window_width 

    vec_replacement = (random_factors[42]*c.storage_TS[store,places.flatten()])+(1-random_factors[42])*ma_vec.flatten()          
    
    c.storage_TS[store,vec_places] = vec_replacement  
    return(c)

@jit(nopython=True)
def storage_specification_operator(c,random_factors,n_store,storage_characteristics,hyperparameters_operators_num):
    stores = np.random.choice(n_store,2,replace=False)
    len_subset = np.random.randint(1,max(3,int(len(c.storage_TS[0])/hyperparameters_operators_num[1,9])))         
    starts = np.random.choice(np.arange(0,len(c.storage_TS[0])-len_subset+1),np.random.randint(1,min(20,int(len(c.storage_TS[0])/hyperparameters_operators_num[6,9])+2)))              
    places=np.empty((len(starts),len_subset),dtype=np.int64)
    subset = np.empty((2,len(starts),len_subset),dtype=np.float64)
    effective_actions = np.empty((2,len(starts),len_subset),dtype=np.float64)
    for i in range(len(starts)):
        places[i]=np.arange(starts[i],starts[i]+len_subset)
        subset[:,i] = c.storage_TS[:,places[i]][stores,:]
        for j in range(2):
           effective_actions[j,i]=np.where(subset[j,i]<0,subset[j,i]*storage_characteristics[4,stores[j]],subset[j,i])
        
    total_actions = effective_actions.sum(axis=0)
    results = total_actions.sum(axis=1)
    trend = results/len_subset
    noise = total_actions-np.repeat(trend,len_subset).reshape(len(starts),len_subset)
    coeff = random_factors[43]
    c.storage_TS[stores[0],places.flatten()] = (1-coeff)*c.storage_TS[stores[0],places.flatten()]+np.repeat(coeff*trend,len_subset)
    c.storage_TS[stores[1],places.flatten()] = (1-coeff)*c.storage_TS[stores[1],places.flatten()]+coeff*noise.flatten()
    return(c)

@jit(nopython=True)
def constraint_operator(c,constraint_num,hyperparameters_operators_num):
    if (constraint_num==1) :
      sample = c.trades>0
      if (sum(sample)>1):
          mutations_TS = np.random.choice(np.where(sample)[0],np.random.randint(1,max(2,int(sum(sample)/hyperparameters_operators_num[1,11]))),replace=False)
          modifs = c.trades[mutations_TS]*np.random.uniform(hyperparameters_operators_num[2,11],hyperparameters_operators_num[3,11])
          c.storage_TS[:,mutations_TS] = c.storage_TS[:,mutations_TS]+modifs
      
          poss_ant = np.where(sample==False)[0]
          if (len(poss_ant)>0):
              anti_mutations = np.random.choice(poss_ant,np.random.randint(1,1+min(len(poss_ant),max(1,int(sum(sample)/hyperparameters_operators_num[1,11])))),replace=False)      
              c.storage_TS[:,anti_mutations]=c.storage_TS[:,anti_mutations]-sum(modifs)/len(anti_mutations)
    return(c)

@jit(nopython=True)
def Y_DSM_operator(c,hyperparameters_operators_num):
    mutations_Y_DSM=np.random.choice(len(c.Y_DSM), size=np.random.randint(1,max(2,int(len(c.Y_DSM)/hyperparameters_operators_num[1,13]))), replace=False)
    c.Y_DSM[mutations_Y_DSM]=np.maximum(0,(c.Y_DSM[mutations_Y_DSM] + np.random.normal(0,50*hyperparameters_operators_num[5,13],len(mutations_Y_DSM)))*np.random.normal(1,hyperparameters_operators_num[5,14],len(mutations_Y_DSM)))
    return(c)

@jit(nopython=True)
def D_DSM_operator(c,D_DSM_indexes,hyperparameters_operators_num):
    mutations_D_DSM=np.random.choice(D_DSM_indexes, size=np.random.randint(1,max(2,int(len(D_DSM_indexes)/hyperparameters_operators_num[1,14]))), replace=False)
    c.D_DSM[mutations_D_DSM]=np.maximum(0,(c.D_DSM[mutations_D_DSM] + np.random.normal(0,50*hyperparameters_operators_num[5,14],c.D_DSM.shape[1]))*np.random.normal(1,hyperparameters_operators_num[5,14],c.D_DSM.shape[1]))
    return(c)

@jit(nopython=True)
def D_DSM_smoothing_operator(c,D_DSM_indexes,random_factors,hyperparameters_operators_num):
    len_day = int(c.D_DSM.shape[1])
    muted_days=np.random.choice(D_DSM_indexes, size=np.random.randint(1,max(2,int(len(D_DSM_indexes)/hyperparameters_operators_num[1,14]))), replace=False)
    window_width = np.int64(min(3,len_day/8))*2
    len_subset = np.random.randint(1,len_day-window_width)              
    start = np.random.choice(np.arange(int(window_width/2),len_day-len_subset-int(window_width/2)))  
    cumsum_mat=np.empty((len(muted_days),len_subset+window_width),dtype=np.float64)
    for i in range(len(muted_days)):
        cumsum_mat[i] = np.cumsum(c.D_DSM[muted_days[i],np.arange(start-int(window_width/2),start+len_subset+int(window_width/2))])             
   
    ma_vec = (cumsum_mat[:,window_width:] - cumsum_mat[:,:-window_width]) / window_width 
    for i in range(len(muted_days)):        
        c.D_DSM[muted_days[i],np.arange(start,start+len_subset)] = (random_factors[44]*c.D_DSM[muted_days[i],np.arange(start,start+len_subset)])+(1-random_factors[44])*ma_vec[i]  
    return(c)

#Apply the pattern of a random reference day to several days based on their proximity
@jit(nopython=True)
def D_DSM_Interdaily_consistency_operator(c,D_DSM_indexes,random_factors,time_resolution):
    n_days = c.D_DSM.shape[0]
    starts_ref = np.random.choice(D_DSM_indexes),0,np.random.randint(1,n_days),np.random.choice(np.array([-1,1]))
    starts = np.arange(starts_ref[0],starts_ref[0]+starts_ref[2]+1)[::(-1)*starts_ref[3]]%n_days            
    distances = np.abs(starts-starts[0])
    max_distances = max(distances)
    for i in range(len(starts)):
        c.D_DSM[starts[i],:] = ((1-random_factors[45]*(1-distances[i]/(2*max_distances)))*c.D_DSM[starts[i],:])+(random_factors[45]*(1-distances[i]/max_distances/2)*c.D_DSM[starts[0],:])
    return(c)

#Orient the D_DSM where the trades are >0. Changes partially compensated by a random storage 
@jit(nopython=True)
def D_DSM_oriented_operator(c,D_DSM_indexes,time_resolution,choices,random_factors,hyperparameters_operators_num):
    store = choices[12]
    mutations_D_DSM=np.random.choice(D_DSM_indexes,size=np.random.randint(1,max(2,int(len(D_DSM_indexes)/hyperparameters_operators_num[1,14]))),replace=False)
    coeffs=[]
    days=[]
    j=0
    for day in mutations_D_DSM:
        
        ante = c.D_DSM[day].copy()
        coeff_app = (-(c.trades[(day*int(time_resolution*24)):((day+1)*int(time_resolution*24))]-max(c.trades[(day*int(time_resolution*24)):((day+1)*int(time_resolution*24))])))**(2*random_factors[46])
        coeffs.append(coeff_app)
        days.append(day)
        c.D_DSM[day]=np.maximum(0,random_factors[47]*c.D_DSM[day]+(1-random_factors[47])*sum(c.D_DSM[day])*coeff_app/(0.1+sum(coeff_app)))
        c.storage_TS[store][(day*int(time_resolution*24)):((day+1)*int(time_resolution*24))]=c.storage_TS[store][(day*int(time_resolution*24)):((day+1)*int(time_resolution*24))]+(c.D_DSM[day]-ante)*random_factors[48]
        j=j+1
    return(c)

@jit(nopython=True)
def Y_DSM_smoothing_operator(c,random_factors,hyperparameters_operators_num):
    window_width = np.int64(min(6,len(c.Y_DSM)/8))*2
    len_subset = np.random.randint(1,min(30,max(3,int(len(c.Y_DSM)/hyperparameters_operators_num[1,10]))))              
    starts = np.random.choice(np.arange(window_width,len(c.Y_DSM)-len_subset-window_width),np.random.randint(1,min(30,int(len(c.Y_DSM)/hyperparameters_operators_num[6,10])+2)))              
    places=np.empty((len(starts),len_subset),dtype=np.int64)
    places_mva=np.empty((len(starts),len_subset+window_width),dtype=np.int64)
    cumsum_mat=np.empty((len(starts),len_subset+window_width),dtype=np.float64)
    for i in range(len(starts)):
        places[i]=np.arange(starts[i],starts[i]+len_subset)
        places_mva[i] = np.arange(starts[i]-window_width/2,starts[i]+len_subset+window_width/2)
        cumsum_mat[i] = np.cumsum(c.Y_DSM[places_mva[i]]) 
    vec_places = places.flatten()    

    ma_vec = (cumsum_mat[:,window_width:] - cumsum_mat[:,:-window_width]) / window_width 

    vec_replacement = (random_factors[49]*c.Y_DSM[places.flatten()])+(1-random_factors[49])*ma_vec.flatten()          
    
    c.Y_DSM[vec_places] = vec_replacement  
    return(c)

#Apply the pattern of a random reference day to several days based on their proximity
@jit(nopython=True)
def Y_DSM_Interdaily_consistency_operator(c,random_factors,n_bits,time_resolution):
    n_days = int(n_bits/(24*time_resolution))
    len_subset = int(time_resolution*24)
    starts_ref = np.random.choice(np.arange(n_days-1)),0,np.random.randint(1,n_days),np.random.choice(np.array([-1,1]))
    starts = np.arange(starts_ref[0],starts_ref[0]+starts_ref[2]+1)[::(-1)*starts_ref[3]]%n_days            
    places=np.empty((len(starts),len_subset),dtype=np.int64)
    for i in range(len(starts)):
        places[i]=np.arange(starts[i]*int(time_resolution*24)+starts_ref[1],starts[i]*int(time_resolution*24)+starts_ref[1]+(len_subset))    
    distances = np.abs(starts-starts[0])
    max_distances = max(distances)
    for i in range(len(starts)):
        c.Y_DSM[places[i]] = ((1-random_factors[50]*(1-distances[i]/(2*max_distances)))*c.Y_DSM[places[i]])+(random_factors[50]*(1-distances[i]/max_distances/2)*c.Y_DSM[places[0]])
    return(c)

#Orient the Y_DSM where the trades are >0. CHanges partially compensated by a random storage 
@jit(nopython=True)
def Y_DSM_oriented_operator(c,choices,random_factors,hyperparameters_operators_num):
    store = choices[13]
    mutations_Y_DSM=np.random.choice(len(c.Y_DSM), size=np.random.randint(2,max(3,int(len(c.Y_DSM)/hyperparameters_operators_num[1,13]))), replace=False)
    ante = c.Y_DSM[mutations_Y_DSM].copy()
    coeff_app = (-(c.trades[mutations_Y_DSM]-max(c.trades[mutations_Y_DSM])))**(2*random_factors[51])
    c.Y_DSM[mutations_Y_DSM]=np.maximum(0,random_factors[52]*c.Y_DSM[mutations_Y_DSM]+(1-random_factors[52])*sum(c.Y_DSM[mutations_Y_DSM])*coeff_app/(0.1+sum(coeff_app)))
    c.storage_TS[store][mutations_Y_DSM]=c.storage_TS[store][mutations_Y_DSM]+(c.Y_DSM[mutations_Y_DSM]-ante)*random_factors[53]
    return(c)

#Smooth the storage actions on a specific daytime for consecutive days
@jit(nopython=True)
def Long_term_consistency_operator(c,choices,random_factors,time_resolution,hyperparameters_operators_num):
    store = choices[14]
    time_span = int(time_resolution*24)
    len_subset = np.random.randint(1,min(30,max(3,int(len(c.storage_TS[0])/time_span))))      
    starts = np.random.choice(np.arange(4*time_span,len(c.storage_TS[0])-time_span*(len_subset+4)),max(1,int(len(c.storage_TS[store]/hyperparameters_operators_num[6,12]))))
        
    places=np.empty((len(starts),len_subset),dtype=np.int64)
    places_mva=np.empty((len(starts),len_subset+4),dtype=np.int64)
    cumsum_mat=np.empty((len(starts),len_subset+4),dtype=np.float64)
    for i in range(len(starts)):
        places[i]=np.arange(starts[i],starts[i]+len_subset*time_span,time_span)    
        places_mva[i]=np.arange(starts[i]-2*time_span,starts[i]+(len_subset+2)*time_span,time_span)          
        cumsum_mat[i] = np.cumsum(c.storage_TS[store,places_mva[i]]) 
    
    
    window_width = 4
    ma_vec = (cumsum_mat[:,window_width:] - cumsum_mat[:,:-window_width]) / window_width 
    replacement = random_factors[54]*c.storage_TS[store,places.flatten()]+(1-random_factors[54])*ma_vec.flatten()
    
    c.storage_TS[store,places.flatten()] = replacement.flatten()
    return(c)

################################### PRO OPERATORS ##########################################
@jit(nopython=True)
def strategy_operator(c):
    if (c.PMS_strategy == 'LF'):
        c.PMS_strategy = 'CC'
    elif (c.PMS_strategy == 'CC'):
        c.PMS_strategy = 'LF'
    return(c)

@jit(nopython=True)
def production_operator_pro(c,Bounds_prod,groups,hyperparameters_operators_num_pro):    
    group = np.random.randint(len(groups))
    mask_positive = c.production_set[groups[group]]>0
    productor = groups[group][mask_positive]
    productor = productor[0] if len(productor)>0 else np.random.choice(groups[group])

    modifier=max(0,min(Bounds_prod[productor],c.production_set[productor]+np.random.randint(min(-2,-int(Bounds_prod[productor]*hyperparameters_operators_num_pro[1,1])),max(3,int(Bounds_prod[productor]*hyperparameters_operators_num_pro[1,1])))))
    c.production_set[productor]=modifier         
    return(c)

@jit(nopython=True)
def discharge_order_operator(c,random_factors):
    c.PMS_discharge_order = np.argsort(random_factors)
    return(c)

@jit(nopython=True)
def energy_use_repartition_DSM_operator(c,hyperparameters_operators_num_pro):
    c.energy_use_repartition_DSM = min(1,max(0,c.energy_use_repartition_DSM*np.random.normal(1,hyperparameters_operators_num_pro[1,4])))
    return(c)

@jit(nopython=True)
def PMS_taking_over_operator(c,random_factors,hyperparameters_operators_num_pro):
    if (random_factors[0]<0.5):
        c.PMS_taking_over[0,:] = np.sort(np.minimum(1,np.maximum(0,c.PMS_taking_over[0,:]*np.random.normal(1,hyperparameters_operators_num_pro[1,5],9)+np.random.normal(0,hyperparameters_operators_num_pro[1,5]/5,9))))
    if (random_factors[1]<0.5):
        c.PMS_taking_over[1,:] = np.sort(np.minimum(1,np.maximum(0,c.PMS_taking_over[1,:]*np.random.normal(1,hyperparameters_operators_num_pro[1,5],9)+np.random.normal(0,hyperparameters_operators_num_pro[1,5]/5,9))))
    return(c)

@jit(nopython=True)
def PMS_D_DSM_min_levels_operator(c,hyperparameters_operators_num_pro):
    c.PMS_D_DSM_min_levels = np.sort(np.minimum(1,np.maximum(0,c.PMS_D_DSM_min_levels*np.random.normal(1,hyperparameters_operators_num_pro[1,6],23)+np.random.normal(0,hyperparameters_operators_num_pro[1,6]/5,23))))
    return(c)

@jit(nopython=True)
def PMS_Y_DSM_min_levels_operator(c,hyperparameters_operators_num_pro):
    c.PMS_Y_DSM_min_levels = np.sort(np.minimum(1,np.maximum(0,c.PMS_Y_DSM_min_levels*np.random.normal(1,hyperparameters_operators_num_pro[1,6],11)+np.random.normal(0,hyperparameters_operators_num_pro[1,6]/5,11))))
    return(c)

@jit(nopython=True)
def PMS_DG_min_runtime_operator(c):
    c.PMS_DG_min_runtime = max(1,c.PMS_DG_min_runtime+np.random.randint(-1,2))
    return(c)

@jit(nopython=True)
def PMS_DG_min_production_operator(c,hyperparameters_operators_num_pro):
    c.PMS_DG_min_production = max(0.,np.exp(np.log(c.PMS_DG_min_production+0.1)*np.random.normal(1,hyperparameters_operators_num_pro[1,8])))
    return(c)

@jit(nopython=True)
def storages_capacity_operator(c,hyperparameters_operators_num_pro,choice):
    c.storages[0,choice] = max(0.,np.exp(np.log(c.storages[0,choice]+0.1)*np.random.normal(1,hyperparameters_operators_num_pro[1,9])))
    return(c)

@jit(nopython=True)
def storages_inpower_operator(c,hyperparameters_operators_num_pro,choice):
    c.storages[1,choice] = max(0.,np.exp(np.log(c.storages[1,choice]+0.1)*np.random.normal(1,hyperparameters_operators_num_pro[1,10])))
    return(c)

@jit(nopython=True)
def storages_outpower_operator(c,hyperparameters_operators_num_pro,choice):
    c.storages[2,choice] = max(0.,np.exp(np.log(c.storages[2,choice]+0.1)*np.random.normal(1,hyperparameters_operators_num_pro[1,10])))
    return(c)

@jit(nopython=True)
def SOC_init_operator(c,random_factor,choice):
    c.storages[3,choice] = max(0.,min(1,c.storages[3,choice]*(0.5+random_factor)))
    return(c)



