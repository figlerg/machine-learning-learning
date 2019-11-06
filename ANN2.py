import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Qt5Agg')
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
from Init import X_normalized,Y,N,M,C,attributeNames

X = X_normalized
y = Y[:,1].reshape(-1,1).astype(float)


complexities = [1,5,8,10,15,20] # these are the no of hidden layers we try
K = 3
n_replicates = 1
max_iter = 10000
nr_of_layers = 5  # includes transfer fcts

CV = model_selection.KFold(K,shuffle=True) # creates class object for k-fold cross validation


# TODO maybe include colors? why?
# TODO had to move model to inner loop so i could change nr of hidden units
errors1 = []  # make a list for storing generalizaition error in each loop
errors2 = [] # TODO base model for comparison

errors_table = np.zeros((K,K))
saved_errors = {} # here I'll save all the generalisation errors in the innermost loop
saved_nets = {} # saving the nets
saved_outer_errors = []

comp_plot = plt.figure()
for (k, (train_index, test_index)) in enumerate(CV.split(X, y)):
    print('\nCrossvalidation fold: {0}/{1}'.format(k + 1, K))

    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.tensor(X[train_index, :], dtype=torch.float)
    y_train = torch.tensor(y[train_index], dtype=torch.float)
    X_test = torch.tensor(X[test_index, :], dtype=torch.float)
    y_test = torch.tensor(y[test_index], dtype=torch.float)

    CV_inner = model_selection.KFold(K, shuffle=True)

    for (i, (train_index_inner, val_index)) in enumerate(CV_inner.split(X_train, y_train)):
        X_train_inner = torch.tensor(X[train_index_inner, :], dtype=torch.float)
        y_train_inner = torch.tensor(y[train_index_inner], dtype=torch.float)
        X_val = torch.tensor(X[val_index, :], dtype=torch.float)
        y_val = torch.tensor(y[val_index], dtype=torch.uint8)

        net = []
        net_counter = 0
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
                                                               X=X_train,
                                                               y=y_train,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
            net_counter +=1
            print('\n\tBest loss: {}\n'.format(final_loss))

            # Determine estimated class labels for test set
            # TODO this is wrong right? no class labels, but estimate

            y_val_est = net(X_val)

            # Determine errors and errors
            se = (y_val_est.float() - y_val.float()) ** 2  # squared error
            mse = (sum(se).type(torch.float) / len(y_val)).data.numpy()  # mean
            # generalisation_error = sum(mse)/len(mse)
            saved_errors[(complexity, i, k)] = mse

    E_i_test = []  # generalization errors
    for complexity in complexities:
        model_comp = np.array([saved_errors[(complexity, i, k)] for i in range(K)])
        E_i_test.append(model_comp.mean())


    plt.plot(complexities,E_i_test)

    selected_complexity = complexities[np.argmin(E_i_test)]

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
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)


    y_test_est = net(X_test)

    se = (y_test_est.float() - y_test.float()) ** 2
    mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()
    saved_outer_errors.append(mse)
    if mse == min(saved_outer_errors):
        best_net = net

print('average loss of models with complexity {0}'.format(selected_complexity) +
      ' is: {0}'.format(np.mean(saved_outer_errors)))






    # # Display the learning curve for the best net in the current fold
    # h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    # h.set_label('CV fold {0}'.format(k + 1))
    # summaries_axes[0].set_xlabel('Iterations')
    # summaries_axes[0].set_xlim((0, max_iter))
    # summaries_axes[0].set_ylabel('Loss')
    # summaries_axes[0].set_title('Learning curves')

print('Diagram of best neural net in outer loop:')
actual_layers = [0,2,4]
weights = [net[i].weight.data.numpy().T for i in actual_layers]
biases = [net[i].bias.data.numpy() for i in actual_layers]
tf = [str(net[i]) for i in actual_layers]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames,
                figsize=(20,20), fontsizes=(10,8))

errors_plt = plt.figure(figsize=(10,10));
y_est = y_test_est.data.numpy(); y_true = y_test.data.numpy();
axis_range = [-5,np.max([y_est, y_true])+1]
plt.plot(axis_range,axis_range,'k--')
plt.plot(y_true, y_est,'ob',alpha=.25)
plt.legend(['Perfect estimation','Model estimations'])
plt.title('Alcohol content: estimated versus true value (for last CV-fold)')
plt.ylim(axis_range); plt.xlim(axis_range)
plt.xlabel('True value')
plt.ylabel('Estimated value')
plt.grid()

plt.show()


print('ANN setup completed')