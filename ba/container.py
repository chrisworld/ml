import sys
import os
import numpy as np
import random
import time
import glob
from scipy.io import loadmat

# load/save data
import cPickle
import pickle
import deepdish as dd
import json
import copy
import scipy.io

import theano

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt    
# ---------------------------------------------------------------------
def plot(x, name='plot', min=-5, max=5, steps=100, show=False):
    
    # set font
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}
    
    matplotlib.rc('font', **font)    
    if show == False: plt.ioff()
    fig = plt.figure()
        
    #plt.imshow(np.flipud(x), interpolation='nearest')
    plt.imshow(x, interpolation='nearest')
    plt.colorbar(boundaries=np.linspace(min,max,steps))
    
    # flip figure
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_title(name)
                
    # save figure        
    plt.savefig(name + '.png')
    if show == False:
        plt.close(fig) 
    else:
        plt.show()


# ------------------------------------------------------------------------------
# MULTIPLE DATA CONTAINER CLASS (CPU)
# ------------------------------------------------------------------------------
"""
     In this a container class for BIG data.
     It returns a loaded data package by the [] operator and
     a shared GPU variable when calling shared()
     
     The class can be used as a list like
     
         for p in MultiContainer:
             process p
     
     where p returns a data package (dict) which is loaded
     on demand. At the same time the shared() will be updated
     as well, that __container__ (CPU) corresponds to 
     __shared_container__ (GPU)
     
     The overall container size can be accessed via size(key)
     
"""
class Container(object):
        
    # --------------------------------------------------------------------------
    def __init__(self, packages, name, normalization='None', noise=False, n_examples=1e9, n_from_each_class=None):
                  
        self.__n_examples = n_examples
        self.__normalization = normalization
        self.__name = name                                     
        self.__packages__ = packages               
        self.__shared_container__ = {}
        self.__idx__ = -1     
        self.__last_loaded__ = -1        
        self.__add_noise = noise
        self.__mean = None
        self.__variance = None
        self.n_packages = len(self.__packages__)                                                         
        self.n_utterances = 0
        self.transcription_map = {}
        self.label_map = {}        
        self.__n_from_each_class = n_from_each_class
                           
        # verbose output
        print('... using %s normalization for %s input' % (normalization, name))
        if noise == True: print('... adding noise to %s input' % (name))
                
        # sanity check
        for p in self.__packages__:
            if not os.path.exists(p):
                print '  given package %s does not exist' % p
                sys.exit() 

    # ---------------------------------------------------------------------
    def blank(self):
        
        if self.label_map.has_key('blank'):
            return self.label_map['blank']
        return -1
    
    # ---------------------------------------------------------------------
    def eof(self):
        
        if self.label_map.has_key('<e>'):
            return self.label_map['<e>']
        return -1        	

    # ---------------------------------------------------------------------
    def __reverse_dict(self, x):
        
        d = {}
        for key,value in x.iteritems():
            d[value] = key
        return d 

    # ---------------------------------------------------------------------
    # flattens a list of lists to a single list
    def __flatten(self, listOflists):
        return reduce(lambda x,y: x+y, listOflists)

    #-------------------------------------------------------------------------------
    def __balance(self, data, targets):
    
        #return data, targets
        if self.__n_from_each_class == None: 
            return data, targets
    
        # fetch argmax of onehot encoded label sequences
        labels = np.argmax(targets, axis=2)
        b = {}     
            
        # iterate over n_utterances
        for i in range(labels.shape[1]):
            # iterate over n_frames 
            for j in range(labels.shape[0]):                         
                label = labels[j,i]                
                if not b.has_key(label): 
                    b[label] = [i]
                elif len(b[label]) < self.__n_from_each_class:
                    b[label].append(i)
                                  
        # define mask for selecting chosen examples
        selected_idx = np.random.permutation(list(set(self.__flatten(b.values()))))
                                                
        # return a unique permuted file list
        return data[:,selected_idx,:], targets[:,selected_idx,:]

    # --------------------------------------------------------------------------
    def __make_blocks_of_N(self, data, targets, weights, batch=50):
                                    
        # stack to multiple of 100 (for easy batch access, by re-using some of the same utterances)
        r = data.shape[1] % batch        
        if r > 0:
	    while True:
                # randomly select some indices
                idx = np.random.permutation(np.arange(data.shape[1]))[:batch-r]
                data = np.concatenate((data, data[:, idx]), axis=1)	
                targets = np.concatenate((targets, targets[:, idx]), axis=1)
                weights = np.concatenate((weights, weights[:, idx]), axis=1)                                            
                r = data.shape[1] % batch        
                if r == 0:
                    self.n_utterances = data.shape[1]        
                    break

        return data, targets, weights
 
    # --------------------------------------------------------------------------
    def add_data(self, data):
        
        self.__intialization_check()          
        data = np.vstack((self.__shared_container__['data'].get_value(), data[0]))        
        self.__shared_container__['data'].set_value(data) 
        labels = np.vstack((self.__shared_container__['labels'].get_value(), data[1]))
        self.__shared_container__['labels'].set_value(labels)
     
    # --------------------------------------------------------------------------
    def add_noise(self, data, noise):
                        
                
        if self.__add_noise == None or noise == None: return data
        print '... adding NOISE'            
        n_frames, n_batch, _ = noise.shape        
        d_frames, d_batch, _ = data.shape
        scale = np.random.uniform(low=0, high=1, size=(1))      
        f_idx = np.random.randint(low=0, high=n_frames-d_frames, size=d_batch)        
        b_idx = np.random.permutation(range(n_batch))        
        for i in range(d_batch):     
            # DEBUG plots           
            #plot(np.transpose(data[:,i]), 'data')            
            #plot(np.transpose(noise[f_idx[i]:f_idx[i]+d_frames,b_idx[i]] * scale), 'noise')                        
            data[:,i] += noise[f_idx[i]:f_idx[i]+d_frames, b_idx[i%n_batch]] * scale
            #plot(np.transpose(data[:,i]), 'mixed')
            #sys.exit()
        
        # re-normalize on utterance level
        mu = np.mean(data,axis=0)
        sigma = np.std(data,axis=0)                      
        data = (data - mu) / (sigma + 1e-9)
        
        return data 

    # --------------------------------------------------------------------------
    def normalize(self, data):
                
        if self.__normalization == 'utterance':            
            # data format <frames, batch, features>            
            mu = np.mean(data,axis=0)
            sigma = np.std(data,axis=0)                    
            data = (data - mu) / (sigma + 1e-9)
        elif self.__normalization == 'standard':            
            data = (data - self.__mean) / self.__variance
        
        return data
    
    # --------------------------------------------------------------------------
    def __rand_roll(self, data, max_shift=100):
                
        print '... applying rand roll (%i) to %s from right -> left' % (max_shift, self.__name)
        data = data.swapaxes(0,1)
        r = -np.random.randint(low=0,high=max_shift,size=data.shape[0])
        rows, column_indices = np.ogrid[:data.shape[0], :data.shape[1]]
        r[r < 0] += data.shape[1]
        column_indices = column_indices - r[:,np.newaxis]
        new_data =  data[rows, column_indices]
        return new_data.swapaxes(0,1)
        print '... done'

    # --------------------------------------------------------------------------
    '''
        adds past N frames to feature dimension (last dim)
        Note: uses np. notation
    '''
    def __transform(self, data, noise=None, rand_roll=False):
        
        # shuffle data if enabled
        #np.random.shuffle(data)
        # add noise if enabled                    
        #data = self.add_noise(data, noise)                
        
        # random roll if enabled
        if rand_roll == True: data = self.__rand_roll(data)
            
        # apply normalization         
        #print '... applying %s normalization to %s' % (self.__normalization, self.__name)
        if self.__normalization == 'utterance':            
            # data format <frames, batch, features>            
            mu = np.mean(data,axis=0)
            sigma = np.std(data,axis=0)                    
            data = (data - mu) / (sigma + 1e-9)
        elif self.__normalization == 'standard':
            data = (data - self.__mean) / self.__variance
     
        return data        
        
    # --------------------------------------------------------------------------
    def select(self, idx):
        
        if idx >= self.n_packages:
            print '... selected package index too high'
            sys.exit()
        self.__idx__  = idx
        self.__update_shared()
        
    # --------------------------------------------------------------------------
    def get_shape(self, key):
        
        self.__intialization_check()
        if not self.__shared_container__.has_key(key):
            print '  key %s does not exists in shared' % key
            sys.exit()                               
        return self.__shared_container__[key].get_value().shape

    # --------------------------------------------------------------------------
    def file(self):
        
        self.__intialization_check()
        return self.__packages__[self.__idx__]
 
    # --------------------------------------------------------------------------
    def mean(self):
        
        self.__intialization_check() 
        return self.__mean
    
    # --------------------------------------------------------------------------
    def variance(self):
        
        self.__intialization_check() 
        return self.__variance
    
    # --------------------------------------------------------------------------
    def values(self):
        
        self.__intialization_check()            
        return self.__shared_container__['data'].get_value(), self.__shared_container__['labels'].get_value()
 
    # --------------------------------------------------------------------------
    def shareds(self):
        
        self.__intialization_check()            
        return self.__shared_container__['data'], self.__shared_container__['labels']
            
    # --------------------------------------------------------------------------
    def next(self):
                        
        if self.__idx__ < self.n_packages - 1:
            self.__idx__ += 1                                                                
            if (self.__last_loaded__ != self.__idx__): self.__update_shared()                    
            return self            
        else:                         
            self.__idx__ = -1    
            raise StopIteration()
    
    # --------------------------------------------------------------------------
    def shuffle():
        
        self.__packages__ = np.random.permutation(self.__packages__)
    
    # --------------------------------------------------------------------------
    def reset(self,):
        self.__idx__ = -1
                
    # --------------------------------------------------------------------------
    def name():
        return self.__name
                
    # --------------------------------------------------------------------------
    def __iter__(self):
        return self
    
    # --------------------------------------------------------------------------
    def __intialization_check(self):
        
        if not self.__shared_container__:
            #print '... creating shared variables for %s' % self.__name
            # load pacakge                        
            package = self.__load(self.__packages__[self.__idx__])
                                                                            
            # set shared            
            self.n_examples = package['data'].shape[1] if self.__n_examples > package['data'].shape[1] else self.__n_examples                                    
            data = package['data'][:, :self.n_examples]            
            labels = package['labels'][:, :self.n_examples]            
             
            # set transcription map
            self.transcription_map = package['label_map']
            self.label_map = self.__reverse_dict(package['label_map'])                    
              
            # load mean & variance 4 normalization            
            self.__mean = package['mean'] if package.has_key('mean') else None
            self.__variance = package['variance']  if package.has_key('variance') else None            
                    
            # update number of utterances
            self.n_utterances = data.shape[1]                                                                                        
            self.transcription_map = package['label_map']
            self.__create_shared('data', self.__transform(data, noise))            
            self.__create_shared('labels', labels)                                               
                
    # --------------------------------------------------------------------------
    def __update_shared(self):
                
        # create shared if not exists        
        self.__intialization_check()
        # update package
        package = self.__load(self.__packages__[self.__idx__])
        # noise
        noise = package['noise'] if self.__add_noise == True else None        
        # apply transformer on input (if enabled)                    
        self.n_examples = package['data'].shape[1] if self.n_examples > package['data'].shape[1] else self.n_examples            
        data = self.__transform(package['data'], noise)[:,:self.n_examples]        
        labels = package['labels'][:,:self.n_examples]        
                        
        # balance if enabled
        data, labels = self.__balance(data, labels)
        
        # make blocks of batch_size shaped data
        data, labels = self.__make_blocks_of_N(data, labels, data_weights)
        
        # add a transcription for the label encoding (neuron numbers do not tell a lot)
        self.transcription_map = package['label_map']
        # load mean & variance 4 normalization
        #self.__mean = package['mean']
        #self.__variance = package['variance']        
        # update number of utterances
        self.n_utterances = data.shape[1]        
        # set shared variable        
        self.__shared_container__['data'].set_value(data)        
        self.__shared_container__['labels'].set_value(labels)        
        self.__last_loaded__ = self.__idx__                                
                    
    # --------------------------------------------------------------------------
    def __create_shared(self, key, value, type='float32'):
                    
        self.__shared_container__[key] = theano.shared(np.asarray(value, type), borrow=True, name='shared' + '_' + self.__name + '_' + key)
    
    # --------------------------------------------------------------------------
    def __load(self, file_path, type_=None):

        
        if type_ == None: 
            type_ = file_path.split('.')[-1]
        obj = None
        #try:
        if True:        
            if type_ == 'matlab':
                obj = scipy.io.loadmat(file_path)
            elif type_ == 'hdf5':
                obj = dd.io.load(file_path)
            elif type_ == 'json':
                f = file(file_path, 'r')
                obj = json.load(f)
                f.close()
            else:
                f = file(file_path, 'r')
                obj = cPickle.load(f)
                f.close()
        #except:
        #    print '... error loading data from file %s' % file_path            

        return obj
                
# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    
    file_list = ['', '', '']
    container = Container(file_list)

