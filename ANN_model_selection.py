# TODO this is the extracted inner part of ANN2. it could work, but i bet some things
#  have to be modified.

# TODO add return values for outer generalization error saving (+ maybe net extraction)

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats


def ANN_model_selection(X_train,y_train,X_test,y_test,K,complexities,M,n_replicates,max_iter):

    saved_errors = {}
    CV_inner = model_selection.KFold(K, shuffle=True)

    for (i, (train_index_inner, val_index)) in enumerate(CV_inner.split(X_train, y_train)):
        # X_train_inner = torch.tensor(X_train[train_index_inner, :], dtype=torch.float)
        # y_train_inner = torch.tensor(y_train[train_index_inner], dtype=torch.float)
        # X_val = torch.tensor(X_train[val_index, :], dtype=torch.float)
        # y_val = torch.tensor(y_train[val_index], dtype=torch.float)

        # since we already expect tensors, this makes more sense:
        X_train_inner = X_train[train_index_inner, :]
        y_train_inner = y_train[train_index_inner]
        X_val = X_train[val_index, :]
        y_val = y_train[val_index]


        for complexity in complexities: # these define the different models
            n_hidden_units = [complexity,complexity]
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, n_hidden_units[0]),  # M features to n_hidden_units
                torch.nn.Tanh(),  # 1st transfer function,
                torch.nn.Linear(n_hidden_units[0], n_hidden_units[1]),  # n1 hidden un. to n2 hidden un.
                torch.nn.Tanh(),  # 2nd transfer function,
                torch.nn.Linear(n_hidden_units[1], 1),  # n2 hidden units to 1 output neuron
                # no final tranfer function, i.e. "linear output"
            )

            # TODO this could be in outer layer as well, but shouldn't change much:
            loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

            # print('Training model of type:\n\n{}\n'.format(str(model())))

            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train_inner,
                                                               y=y_train_inner.unsqueeze(1),
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
            print('\n\tBest loss: {}\n'.format(final_loss))

            # Determine estimated class labels for test set
            # TODO this is wrong right? no class labels, but estimate

            y_val_est = net(X_val).squeeze()

            # Determine errors and errors
            se = (y_val_est.float() - y_val.float()) ** 2  # squared error
            mse = (sum(se).type(torch.float) / len(y_val)).data.numpy()  # mean
            # generalisation_error = sum(mse)/len(mse)
            saved_errors[(complexity, i)] = mse

    E_gen_s = []  # generalization errors
    for complexity in complexities:
        # model_comp = np.array([saved_errors[(complexity, i, k)] for i in range(K)])
        ## k is useless?
        model_comp = np.array([saved_errors[(complexity, i)] for i in range(K)])

        gen_error = model_comp.mean()/K
        E_gen_s.append(gen_error)


    plt.plot(complexities,E_gen_s)

    selected_complexity = complexities[np.argmin(E_gen_s)]

    # repeat training for actual training set
    n_hidden_units=[selected_complexity,selected_complexity]
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units[0]),  # M features to n_hidden_units
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(n_hidden_units[0], n_hidden_units[1]),  # n1 hidden un. to n2 hidden un.
        torch.nn.Tanh(),  # 2nd transfer function,
        torch.nn.Linear(n_hidden_units[1], 1),  # n2 hidden units to 1 output neuron
        # no final tranfer function, i.e. "linear output"
    )

    # TODO this could be in outer layer as well, but shouldn't change much:
    loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

    # print('Training model of type:\n\n{}\n'.format(str(model())))

    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train.unsqueeze(1),
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)


    y_test_est = net(X_test).squeeze()

    se = (y_test_est.float() - y_test.float()) ** 2
    E_i_test = float((sum(se).type(torch.float) / len(y_test)).data.numpy())
    # saved_outer_errors.append(mse)
    # if mse == min(saved_outer_errors):
    #     best_net = net

    return selected_complexity, E_i_test