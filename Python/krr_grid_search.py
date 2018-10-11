from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy import *
import fitsio
from sklearn.externals import joblib

print('Importing BGS data')
BGS_true = fitsio.FITS('/home/sd.lobo251/Documents/dc17b/BGS_files/BGS_true_file.fits')
BGS_tar = fitsio.FITS('/home/sd.lobo251/Documents/dc17b/BGS_files/BGS_tar_file.fits')


print('creating the fuxes array')
alpha = BGS_true[1]['TRUEZ'][:]/BGS_tar[1]['Z'][:]
alpha_sign = alpha>0

from numpy.lib.recfunctions import append_fields
fluxes = BGS_tar[1]['TARGETID','FLUX_G', 'FLUX_R', 'FLUX_Z', \
                                  'FLUX_W1', 'FLUX_W2', 'MW_TRANSMISSION_G', \
                                 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z', \
                                 'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2','Z'][:]

fluxes = append_fields(fluxes, 'TRUEZ', BGS_true[1]['TRUEZ'][:], usemask=False)
fluxes = append_fields(fluxes, 'alpha', alpha, usemask=False)

#Se eliminan los alphas negativos. 
fluxes = fluxes[alpha_sign]
alpha = alpha[alpha_sign]

flux_names = ['FLUX_G', 'FLUX_R', 'FLUX_Z',  'FLUX_W1', 'FLUX_W2']

#changing flux por log(flux)
for name in flux_names:
    fluxes[name] = log10(fluxes[name])
    

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


X_entrada, dict_of_names = transform_to_array(fluxes[0:500000][['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2',  'Z', 'TRUEZ' ]])
y_entrada = fluxes[0:500000]['alpha']

z_col_idx = dict_of_names['Z']
trueZ_col_idx = dict_of_names['TRUEZ']

X_train_i, X_test_i, y_train, y_test = train_test_split(X_entrada, y_entrada , test_size = 0.3, random_state=0)

Z_train = X_train_i[:, [z_col_idx, trueZ_col_idx]]
Z_test = X_test_i[:, [z_col_idx, trueZ_col_idx]]

X_train = omit_columns_of_array([z_col_idx, trueZ_col_idx], X_train_i)
X_test = omit_columns_of_array([z_col_idx, trueZ_col_idx], X_test_i)

#scale
#y_train = reshape(y_train, (len(y_train), 1))
#y_test = reshape(y_test, (len(y_test), 1))
print('Scaling data \n')
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
 

scores = ['r2']#, 'neg_mean_squared_error']
best_parameters = {'r2':0, 'neg_mean_squared_error':0, 'explained_variance':0}
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=3,
                  param_grid={"alpha": [1e0, 0.1],
                              "gamma": logspace(-2, 2, 3)}, n_jobs=8, refit=True, verbose=1)
    
    clf.fit(X_train, y_train)

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

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(r2_score(y_true, y_pred))
    print()
    
best_model = clf.best_estimator_
joblib.dump(best_model, 'best_krr.pkl', compress=1)

y_pred = best_model.predict(X_test)
print('converting to TRUEZ - testing')
predicted_redshifts_test = multiply(y_pred, Z_test[:, 0])
print('converting to TRUEZ - training')
predicted_redshifts_train = multiply(best_model.predict(X_train), Z_train[:, 0])
savetxt('krr_train.out', (predicted_redshifts_train, Z_train[:, 1]))
savetxt('krr_test.out', (predicted_redshifts_test, Z_test[:, 1]))
print('-------- Done ---------')


