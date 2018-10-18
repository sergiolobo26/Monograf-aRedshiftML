#!/usr/bin/env python

from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from numpy.lib.recfunctions import append_fields
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy import *
from sklearn.metrics import r2_score
from astropy.table import Table
from sklearn.utils import shuffle
import time
import argparse

from sklearn.externals import joblib

def import_data(true_path, tar_path):
    
    BGS_true = Table.read(true_path)
    BGS_tar = Table.read(tar_path)
    
    alpha = array(BGS_true['TRUEZ']/BGS_tar['Z'])
    alpha_sign = alpha>0
    
    fluxes = array(BGS_tar['TARGETID','FLUX_G', 'FLUX_R', 'FLUX_Z', \
                                  'FLUX_W1', 'FLUX_W2', 'MW_TRANSMISSION_G', \
                                 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z', \
                                 'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2','Z'])
    
    fluxes = append_fields(fluxes, 'TRUEZ', array(BGS_true['TRUEZ']), usemask=False)
    fluxes = append_fields(fluxes, 'alpha', alpha, usemask=False)
    
    #Se eliminan los alphas negativos. 
    fluxes = fluxes[alpha_sign]
    return fluxes

def transform_to_array(rec_array):
    
    dim_1 = len(rec_array)
    dim_2 = len(rec_array[0])
    temp = zeros((dim_1, dim_2))

    for i in range(dim_1):
        for j in range(dim_2):
            temp[i, j] = rec_array[i][j]
        
    dict_of_names = {name:i  for i, name in enumerate(rec_array.dtype.names)} #index of the names in the new array
    return temp, dict_of_names

def omit_columns_of_array(columns_to_omit, x_array):
    
    dim_1 = x_array.shape[0] #rows
    dim_2 = x_array.shape[1] #columns
    
    n = arange(dim_2)
    
    for i in columns_to_omit:
        msk = n != i
        n = n[msk]
    return x_array[:, n]


def train_test_data(test_size, fluxes_array, n_datos):
    
    fluxes_array = shuffle(fluxes_array, random_state=0)
    
    X_entrada, dict_of_names = transform_to_array(fluxes_array[0:n_datos][['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2',  'Z']])
    y_entrada = fluxes_array[0:n_datos]['TRUEZ']

    z_col_idx = dict_of_names['Z']

    X_train, X_test, y_train, y_test = train_test_split(X_entrada, y_entrada , test_size = test_size, random_state=0)

    Z_tar_train = reshape(X_train[:, [z_col_idx]], (len(X_train)))
    Z_tar_test = reshape(X_test[:, [z_col_idx]], (len(X_test)))
    
    return X_train, X_test, y_train, y_test, Z_tar_train, Z_tar_test

def Standard_scale_data(X_train, X_test):
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    return X_train, X_test


def main():
    '''
    '''
    
    parser = argparse.ArgumentParser(description='n_jobs to fit the model')
    parser.add_argument('--n_jobs', help="number of procesors to fit the model", type= int, default=1)
    arguments = parser.parse_args()
    n_jobs = arguments.n_jobs
    
    total_time = time.time()
    print()
    print('Importing BGS data, generating the fluxes array')
    print()
    true_data_path = '/hpcfs/home/sd.lobo251/Redshift_ML/BGS_files/BGS_true_file.fits'
    tar_data_path = '/hpcfs/home/sd.lobo251/Redshift_ML/BGS_files/BGS_tar_file.fits'
    fluxes = import_data(true_data_path, tar_data_path)
    fluxes
    flux_names = ['FLUX_G', 'FLUX_R', 'FLUX_Z',  'FLUX_W1', 'FLUX_W2']

    #changing flux por log(flux)
    print()
    print('computing Log Flux, Log Alpha')
    print()
    for name in flux_names:
        fluxes[name] = log10(fluxes[name])
    fluxes['alpha'] = log10(fluxes['alpha'])

    n_data_points = [20000]
    fit_time_per_n = [[] for i in range(len(n_data_points))]
    predict_time_per_n = [[] for i in range(len(n_data_points))]
    n_repetitions = 4
    for i in range(n_repetitions):
        for j, n_data in enumerate(n_data_points):

            print('Generating train-test set for {} data points \n'.format(n_data))
            X_train, X_test, y_train, y_test, Z_tar_train, Z_tar_test = train_test_data(test_size=0.2, fluxes_array=fluxes, n_datos=n_data)

            print('Scaling data \n')

            X_train, X_test = Standard_scale_data(X_train, X_test)

            score = 'r2'#, 'neg_mean_squared_error']
            best_parameters = {'r2':0, 'neg_mean_squared_error':0, 'explained_variance':0}
            print("# Tuning hyper-parameters for %s" % score)
            print()
            clf = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=3, n_jobs=n_jobs,
                      param_grid={"alpha": [0.01, 0.001, 0.0001],
                                  "gamma": [0.1, 1], 'kernel':['rbf']}, refit=True)

            print('Fitting model with n_jobs = {}'.format(n_jobs))
            print()
            t0 = time.time()
            clf.fit(X_train, y_train)
            fit_time = time.time() - t0
            fit_time_per_n[j].append(fit_time)
            print()
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            best_parameters[score] = clf.best_params_
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
            print()
            print("%s score computed on the full evaluation set:" % score)
            print()

            t0 = time.time()
            y_true, y_pred = y_test, clf.predict(X_test)
            predict_time = time.time() - t0
            predict_time_per_n[j].append(predict_time)
            model_name = 'krr_model_{}'.format(i)
            best_model = clf.best_estimator_
            print(r2_score(y_true, y_pred))
            print()
            print('Saving Model to {}'.format(model_name))
            joblib.dump(best_model, '{}.pkl'.format(model_name), compress=1)
            y_train_pred = best_model.predict(X_train)
            savetxt('{}.txt'.format(model_name), (y_train_pred, y_train, Z_tar_train))
            savetxt('{}.txt'.format(model_name), (y_pred, y_test, Z_tar_test))
            print('-----Done for n = {}-----'.format(n_data))
    
    #best_model = clf.best_estimator_
    #joblib.dump(best_model, 'best_krr.pkl', compress=1)
    total_run_time = time.time() - total_time
    #y_pred = best_model.predict(X_test)
    #y_train_pred = best_model.predict(X_train)
    #savetxt('krr_train.out', (y_train_pred, y_train, Z_tar_train))
    #savetxt('krr_test.out', (y_pred, y_test, Z_tar_test))
    set_printoptions(precision=4)
    print()
    print('-----------------------------------Results--------------------------------------------')
    print('Time results after {} repetitions, n_jobs = {}'.format(n_repetitions, n_jobs))
    print('n_datos: ', n_data_points)
    print()
    print('fit_time mean: ', array(fit_time_per_n).mean(axis=1))
    print('fit_time error: ', array(fit_time_per_n).std(axis=1)/sqrt(n_repetitions), '\n')

    print('predict_time mean: ', array(predict_time_per_n).mean(axis=1))
    print('predict_time error: ', array(predict_time_per_n).std(axis=1)/sqrt(n_repetitions), '\n')
    print('Total run time: {:.2f} s'.format(total_run_time))
    print('------------------------------------ Done --------------------------------------------')

if __name__ == '__main__':
    main()

