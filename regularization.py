from Init import *
from sklearn import model_selection
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
from toolbox_02450 import rlr_validate
from ANN_model_selection import ANN_model_selection
import torch

y_reg = np.asarray(raw_data[:,11]).astype(np.float64)
X_reg = np.c_[autumn, spring, summer, winter, large_river, medium_river, small_river, high_flow, low_flow, medium_flow, X_num]
X_reg = np.concatenate((np.ones((X_reg.shape[0],1)),X_reg),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

X1coeff=np.ones((X_reg.shape[0],1))

K = 10
CV = model_selection.KFold(K, shuffle=True)

lambdas = np.power(10.,range(-5,9))

Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
Error_train_1coeff = np.empty((K,1))
Error_test_1coeff = np.empty((K,1))

Error_test_ann = np.empty((K,1))

optimal_lambda = np.empty((K,1))
optimal_complexity = np.empty((K,1))



w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

k=0
for train_index, test_index in CV.split(X_reg,y_reg):
    
    # extract training and test set for current CV fold
    X_train = X_reg[train_index]
    y_train = y_reg[train_index]
    X_test = X_reg[test_index]
    y_test = y_reg[test_index]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float)

    
    #creation of the baseline
    X_train_1coeff = X1coeff[train_index]
    X_test_1coeff = X1coeff[test_index]
    
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    #standardization
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    #regression on the baseline
    reg_1coeff = lm.LinearRegression(fit_intercept=True).fit(X_train_1coeff, y_train)
    Error_train_1coeff[k] = np.square(y_train-reg_1coeff.predict(X_train_1coeff)).sum()/y_train.shape[0]
    Error_test_1coeff[k] = np.square(y_test-reg_1coeff.predict(X_test_1coeff)).sum()/y_test.shape[0]
    
    
    # Estimate weights for the optimal value of lambda, on entire training set
    optimal_lambda[k]=opt_lambda
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    ##
    # ADDED BY FELIX:
    complexities = [1,5,8,9,10,15] # based on a few test runs
    n_replicates = 1
    max_iter =10000

    # self written ann model selection in separate file
    opt_complexity, E_i_test = ANN_model_selection(X_train_tensor,y_train_tensor,
                                                   X_test_tensor,y_test_tensor,
                                                   K,complexities,M,n_replicates,max_iter)
    optimal_complexity[k] = opt_complexity
    Error_test_ann[k] = E_i_test

    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-')
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()

    k+=1

print('Optimal lambda', optimal_lambda)
print('Error baseline', np.round(Error_test_1coeff,2))
print('Error regularized regression', np.round(Error_test_rlr,2))
    
show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))
