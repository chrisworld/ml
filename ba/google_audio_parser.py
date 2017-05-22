import sys
import os
import numpy as np
import glob
from scipy.io import loadmat

# load/save data
import cPickle
import pickle
import json
import deepdish as dd
import copy
import scipy.io
import scipy.io.wavfile as wav

from features_extraction.features.base import spectrogram
from features_extraction.features.base import mfcc
from container import Container


# ------------------------------------------------------------------------------
# GOOGLE_AUDIO LOADER CLASS
# ------------------------------------------------------------------------------
"""
     This is a parser class for the GOOGLE_AUDIO data
"""

class GOOGLE_AUDIO(object):

    # --------------------------------------------------------------------------
    def __init__(self, config={}):
                        
        # default parameter initialization
        self.__default(config, 'name', 'GOOGLE_AUDIO')                        
        self.__default(config, 'normalization', 'utterance')                
        #self.__default(config, 'n_utterances', { 'train': 1000, 'test': 1000, 'unlabeled':1000 })        
        self.__default(config, 'n_utterances', { 'train': int(1e9), 'test': int(1e9), 'unlabeled':int(1e9) })
        self.__default(config, 'package_size', 5000) # 5000 utterances per package                                    
        self.__default(config, 'ignore_labels', []) # maximum target length                     
        self.__default(config, 'overwrite', False) # overwrite cache                
                                  
        print '... using GOOGLE_AUDIO dataset, containing up to %i spoken labels per utterance' % config['max_target_len']
                
        # mean & variance (this is a scalar value after computation)
        #self.__mean = { 'data':[], 'labels':[] }
        #self.__variance = { 'data':[], 'labels':[] }        
           
        # storage variables
        self.config = config      
        self.name = 'GOOGLE_AUDIO'
        self.__label_map = {}    
                                    
        # define paths        
        tag = 'frame_level'        
        self.cache_path_ = {'train':'./data/google_audio/' + config['name'] + '/%s/*train_%s_%s.hdf5' % (tag, config['features'], config['encoding']),  
                            'test':'./data/google_audio/' + config['name'] + '/%s/*test_%s_%s.hdf5'  % (tag, config['features'], config['encoding'])}
                        
        self.data_path = '/afs/spsc.tugraz.at/resources/databases_new/all/timit/'  # EDIT TO GOOGLE PATH                                      
        self.train_file_paths = self.data_path + '/timit/train/*/*/*.trecord' # EDIT!!! PATH to tfrecord                                   
        self.test_file_paths = self.data_path + '/timit/test/*/*/*.trecord' # EDIT!!! PATH to tfrecord
           

    # --------------------------------------------------------------------------
    def __default(self, config, key, value):
        
        if not config.has_key(key):
            config[key] = value
        return config[key]

    # --------------------------------------------------------------------------
    def cache_path(self):                   
        return os.path.normpath(self.cache_path_)

    #----------------------------------------------------------------------------
    def __load_cached(self):
                
        # overwrite cache
        if self.config['overwrite']: 
            return None
                     
        # load cached version        
        results = {}
        unlabeled = []    
        for set_ in self.cache_path_.keys():                      
            file_paths = glob.glob(self.cache_path_[set_])            
            if len(file_paths) > self.config['n_packages']:
                file_paths = file_paths[:self.config['n_packages']]
                print '... restricting %s to %i packages' % (set_, self.config['n_packages'])                
            num_files = len(file_paths)                                                                                   
            if not num_files:                 
                return None            
            else:                                
                print '... loading cached %s data (%i packages) (%s)' % (set_, len(file_paths), file_paths[-1])
                n_from_each_class = self.config['n_from_each_class'] if set_ == 'train' else None                   
                results[set_] = Container(file_paths, set_, normalization=self.config['normalization'], n_examples=self.config['n_utterances'][set_], n_from_each_class=n_from_each_class, noise=self.config['noise'])
                if self.config['unlabeled_data_set'] == 'all':
                    print '... using %s as unlabeled dataset' % set_
                    unlabeled.extend(file_paths)
                elif set_ == self.config['unlabeled_data_set']:
                    print '... using %s as unlabeled dataset' % set_ 
                    unlabeled.extend(file_paths)                                                                   
                                        
        # return cached version if exists            
        return results
       
    # --------------------------------------------------------------------------
    def load(self):
            
        #----------------------------------------------------------------------------
        # load cached version
        #----------------------------------------------------------------------------
        packages = self.__load_cached()
        if packages != None:         
            return packages
        
        #----------------------------------------------------------------------------
        # generate data packages
        #----------------------------------------------------------------------------
        print '... no cached files found %s, generating data packages' % self.cache_path_['train']
        package_size = self.config['package_size']                                                                 
        train_file_paths = glob.glob(self.train_file_paths)       
        num_train_files = len(train_file_paths)                                                                                     
        test_file_paths = glob.glob(self.test_file_paths)
        num_test_files = len(test_file_paths)         

        # sanity check
        if not num_train_files: print '... error, no train data files found in %s' % self.train_data_file_path        
        if not num_test_files: print '... error, no test data files found in %s' % self.test_data_file_path
                   
        #----------------------------------------------------------------------------
        # parse .mat files / each source ['bus','ped', 'str', 'caf'] from storage location 
        # and store all utterances in data packages of size step_size                
        #----------------------------------------------------------------------------        
        '''
            PROCESS TRAIN DATA
        '''                                                        
        print '... processing %i %s files' % (self.config['n_utterances']['train'], 'train')                    
        train_package_paths = self.__create_packages(train_file_paths, 'train')                          
                                                     
        '''
            PROCESS TEST DATA
        '''                                      
        print '... processing %i %s files' % (self.config['n_utterances']['test'], 'test')
        test_package_paths = self.__create_packages(test_file_paths, 'test')                
                                                                                   
        #----------------------------------------------------------------------------
        # store normalization pararmeters (normaliation is applied on the fly)
        #----------------------------------------------------------------------------        
        #mean, std = self.__compute_normalizations(train_package_paths)        
        #self.__compute_normalizations(test_package_paths, mean, std)
        
        #----------------------------------------------------------------------------            
        # create data container which handle multiple data packages (load on demand for BIG DATA on GPU)
        #----------------------------------------------------------------------------        
        result = {'train':Container(train_package_paths, 'train', normalization=self.config['normalization']), #, n_examples=self.config['n_utterances']['train']), 
                  'test':Container(test_package_paths, 'test', normalization=self.config['normalization']), #, n_examples=self.config['n_utterances']['test']),
                 } 
                                     
        return result

    # --------------------------------------------------------------------------
    def __create_packages(self, file_list, set_):
        
      
        # TODO
        # create label map
        self.__label_map = load_csv()
      
        for n, f in enumerate(file_list):
            
            if u > self.config['n_utterances'][set_] - 1: 
                break

            # TODO            
            x,y = load_TF_record(f)          
            
             
                                                                        
            # add to container
            features.extend(x)
            labels.extend(y)
                   
        # verbose output
        print '... creating data containers for set %s' % set_
              
        # save to container format
        X = []
        Y = []        
        for x, y in zip(features, labels):

            X.append(x)
            Y.append(y)
            count = count + 1  
                                                                                                             
            # save package 
            if count >= self.config['package_size']:
                                                
                # add to conainer                                                                            
                data = {'data':   np.asarray(X, 'float32').swapaxes(1,0),
                        'labels': np.asarray(Y, 'float32').swapaxes(1,0),                        
                        'label_map': self.labels()}
                                    
                train_package_path = self.cache_path_[set_].replace('*%s' % set_, 'package_' + str(step) + '_%s' % set_)
                print '... saving %s' % train_package_path            
                self.__save(data, train_package_path)  # save data package
                
                # reset lists
                X = []
                Y = []
                
                # package counter
                step += 1
                count = 0
                
                # package paths
                package_paths.append(train_package_path)                                
              
        # write the last package                
        if count > 0:            
            # add to conainer                                                                            
            data = {'data':   np.asarray(X, 'float32').swapaxes(1,0),
                    'labels': np.asarray(Y, 'float32').swapaxes(1,0),                    
                    'label_map': self.labels()}
                       
            package_path = self.cache_path_[set_].replace('*%s' % set_, 'package_' + str(step) + '_%s' % set_)
            print '... saving %s' % package_path
            self.__save(data, package_path)  # save data package            
            package_paths.append(package_path)
       
        return package_paths                                                                  
                                               
    # --------------------------------------------------------------------------
    def __save(self, obj, file_path, type_='hdf5'):

        # sanity check
        if file_path == '':
            return

        # sanity check
        path = os.path.dirname(file_path)
        if not os.path.exists(path):
            os.makedirs(path)

        # save obj
        if type_ == 'matlab':
            scipy.io.savemat(file_path, obj)
        elif type_ == 'hdf5':
            dd.io.save(file_path, obj, compression='default')                                    
        elif type_ == 'json':
            f = file(file_path, 'wb')
            s = json.dumps(obj, indent=0)
            for i in xrange(len(s)):
                f.write(s[i])
            f.write('\n')
            f.close()
        else:
            f = file(file_path, 'wb')
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

    # --------------------------------------------------------------------------
    def __load(self, file_path, type_='hdf5'):

        obj = None
        try:        
            if type_ == 'matlab':
                # does not work with matlab >=7.3
                obj = scipy.io.loadmat(file_path)
                #obj = h5py.File(file_path)
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
        except:
            print '... error loading data from file %s' % file_path
            return None

        return obj
    
# ---------------------------------------------------------------------------------
if __name__ == '__main__':

    parser = GOOGLE_AUDIO({'overwrite':True})
    packages = parser.load()
    
    # tests the gpu shared iterator
    print '    extraction completed'
    print '    testing iterators ...'    
    for set_ in packages.keys():
        for i, p in enumerate(packages[set_]):                
            inputs, labels = p.values()                        
            print('       %s input:  %s [frames, batch, features] [min: %s, max: %s] (%s packages)' % (set_.ljust(10), str(inputs.shape).ljust(15), str('%.2f' % np.min(inputs)).ljust(4), str('%.2f' % np.max(inputs)).ljust(4), str(packages[set_].n_packages)))
            print('       %s labels: %s [frames, batch, labels]   [min: %s, max: %s] (%s packages)' % (set_.ljust(10), str(labels.shape).ljust(15), str('%.2f' % np.min(labels)).ljust(4), str('%.2f' % np.max(labels)).ljust(4), str(packages[set_].n_packages)))             
    print '    label map: %s' % packages['train'].label_map
