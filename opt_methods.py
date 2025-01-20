import torch
import numpy as np


# Function to compute the gradient of a single component function f_i with respect to x
def compute_grad_i(f, x, i):
    """
    Computes the gradient of a single component function f_i with respect to x.

    Inputs:
        f: The objective function
        x: Input tensor
        i: Index for the component function

    Outputs:
        grad: Gradient tensor of the function f_i
    """
    x = x.clone().detach().requires_grad_(True)
    y_hat = f(x, i)
    y_hat.backward()
    grad = x.grad.clone()
    x.grad.zero_()
    return grad


# Function to compute the gradient of the overall objective function with respect to x
def compute_grad(f, x):
    """
    Computes the gradient of the objective function with respect to x.

    Inputs:
        f: The objective function
        x: Input tensor

    Outputs:
        grad: Gradient tensor of the function f
    """
    x = x.clone().detach().requires_grad_(True)
    y_hat = f(x)
    y_hat.backward()
    grad = x.grad.clone()
    x.grad.zero_()
    return grad


# Gradient descent implementation
def gd(problem, stepsize, n_iter=100, verbose=False):
    """
    Performs gradient descent optimization.

    Inputs:
        problem: Problem instance containing loss and objective functions
        stepsize: Constant step size for gradient updates
        n_iter: Number of iterations
        verbose: If True, prints progress at each iteration

    Outputs:
        objvals: Numpy array containing the history of objective values
    """
    objvals = []  # Objective value history
    x = problem.x.clone().detach()  # Initial iterate
    obj = problem.obj_func(x).detach()  # Initial objective value
    g = compute_grad(problem.loss, x)  # Initial gradient
    objvals.append(obj)

    if verbose:
        print("Gradient Descent:")
        print(' | '.join([name.center(8) for name in ["iter", "fval"]]))
        print(' | '.join([("%d" % 0).rjust(8), ("%.2e" % obj).rjust(8)]))

    for k in range(n_iter):
        # Gradient descent update
        x = x - stepsize * g
        g = compute_grad(problem.loss, x)  # Recompute gradient
        obj = problem.obj_func(x).detach()  # Compute new objective
        objvals.append(obj)

        if verbose:
            print(' | '.join([("%d" % (k + 1)).rjust(8), ("%.2e" % obj).rjust(8)]))

    problem.x = x.clone().detach()  # Update problem with the final iterate
    return np.array(objvals)


# Stochastic gradient implementation
def stoch_grad(problem, stepchoice=0, step0=1, n_epoch = None, n_iter=1000, nb=1, with_replace=False, verbose=True):
    """
    Performs stochastic gradient descent optimization.

    Inputs:
        problem: Problem instance containing loss and objective functions
        stepchoice: Step size selection strategy
            0: Constant step size of 1/L
            t > 0: Decreasing step size 1/(k+1)^t
        step0: Initial step size
        n_iter: Number of iterations
        nb: Batch size for stochastic updates
        with_replace: If True, samples components with replacement
        verbose: If True, prints progress at each iteration

    Outputs:
        x_output: Final iterate of the method
        objvals: History of objective values
        normits: History of iterate norms
    """
    objvals = []  # Objective value history
    normits = []  # Norm of iterates history
    L = problem.lipgrad()  # Lipschitz constant
    n = problem.n  # Number of components
    x = problem.x.clone().detach()  # Initial iterate

    obj = problem.obj_func(x).detach()  # Initial objective value
    objvals.append(obj)
    nx = torch.norm(x)  # Norm of the iterate
    normits.append(nx)

    if verbose:
        print("Stochastic Gradient, batch size =", nb, "/", n)
        print(' | '.join([name.center(8) for name in ["iter", "fval", "normit"]]))
        print(' | '.join([("%d" % 0).rjust(8), ("%.2e" % obj).rjust(8), ("%.2e" % nx).rjust(8)]))

    if n_epoch is not None:
        n_iter = n_epoch * (n // nb)

    for k in range(n_iter):
        # Draw batch indices
        ik = np.random.choice(n, nb, replace=with_replace)

        # Compute batch gradient
        sg = torch.zeros_like(x)
        for j in range(nb):
            gi = compute_grad_i(problem.obj_func_i, x, ik[j])
            sg += gi
        sg /= nb  # Average gradient over the batch

        # Update step size and iterate
        if stepchoice == 0:
            x -= (step0 / L) * sg
        elif stepchoice > 0:
            sk = float(step0 / ((k + 1) ** stepchoice))
            x -= sk * sg

        nx = torch.norm(x)  # Update norm of the iterate
        obj = problem.obj_func(x).detach()  # Update objective value

        objvals.append(obj)
        normits.append(nx)

        if verbose and (k + 1) % (n // nb) == 0:
            print(' | '.join([("%d" % (k + 1)).rjust(8), ("%.2e" % obj).rjust(8), ("%.2e" % nx).rjust(8)]))

    return x.clone().detach(), np.array(objvals), np.array(normits)

def adagrad(problem,xtarget,stepchoice=0,step0=1, n_epoch = None, n_iter=1000,nb=1,average=0,scaling=0,with_replace=False,verbose=False):
    """
        A code for gradient descent with various step choices.

        Inputs:
            x0: Initial vector
            problem: Problem structure
                problem.fun() returns the objective function, which is assumed to be a finite sum of functions
                problem.n returns the number of components in the finite sum
                problem.grad_i() returns the gradient of a single component f_i
                problem.lipgrad() returns the Lipschitz constant for the gradient
                problem.cvxval() returns the strong convexity constant
                problem.lambda returns the value of the regularization parameter
            xtarget: Target minimum (unknown in practice!)
            stepchoice: Strategy for computing the stepsize
                0: Constant step size equal to 1/L
                t>0: Step size decreasing in 1/(k+1)^t
            step0: Initial steplength (only used when stepchoice is not 0)
            n_iter: Number of iterations, used as stopping criterion
            nb: Number of components drawn per iteration/Batch size
                1: Classical stochastic gradient algorithm (default value)
                problem.n: Classical gradient descent (default value)
            average: Indicates whether the method computes the average of the iterates
                0: No averaging (default)
                1: With averaging
            scaling: Use a diagonal scaling
                0: No scaling (default)
                (0,1): Average of magnitudes (RMSProp)
                1: Normalization with magnitudes (Adagrad)
            with_replace: Boolean indicating whether components are drawn with or without replacement
                True: Components drawn with replacement
                False: Components drawn without replacement (Default)
            verbose: Boolean indicating whether information should be plot at every iteration (Default: False)

        Outputs:
            x_output: Final iterate of the method (or average if average=1)
            objvals: History of function values (Numpy array of length n_iter at most)
            normits: History of distances between iterates and optimum (Numpy array of length n_iter at most)
    """
    ############
    # Initial step: Compute and plot some initial quantities

    # objective history
    objvals = []

    # iterates distance to the minimum history
    normits = []

    # Lipschitz constant
    L = problem.lipgrad()

    # Number of samples
    n = problem.n

    # Initial value of current iterate
    x = problem.x.clone().detach()
    nx = torch.norm(x)

    # Average (if needed)
    if average:
            xavg=np.zeros(len(x))

    #Scaling values
    if scaling>0:
        eps=1/(2 *(n ** (0.5))) # To avoid numerical issues
        v = torch.zeros(problem.d)

    # Initialize iteration counter
    k=0

    # Current objective
    obj = problem.obj_func(x)
    objvals.append(obj)
    # Current distance to the optimum
    nmin = torch.norm(x-xtarget)
    normits.append(nmin)

    # Plot initial quantities of interest
    if verbose:
        print("Stochastic Gradient, batch size=",nb,"/",n)
        print(' | '.join([name.center(8) for name in ["iter", "fval", "normit"]]))
        print(' | '.join([("%d" % k).rjust(8),("%.2e" % obj).rjust(8),("%.2e" % nmin).rjust(8)]))

    if n_epoch is not None:
        n_iter = n_epoch*(n//nb)

    ################
    # Main loop
    while (k < n_iter and nx < 1e8):

        ik = np.random.choice(n,nb,replace=with_replace)# Batch gradient
        # Stochastic gradient calculation
        sg = torch.zeros(problem.d)
        for j in range(nb):
            gi = compute_grad_i(problem.obj_func_i, x, ik[j])
            sg += gi
        sg = (1/nb)*sg

        if scaling>0:
            if scaling==1:
                # Adagrad update
                v = v + sg*sg
            else:
                # RMSProp update
                v = scaling*v + (1-scaling)*sg*sg
            sg = sg/(np.sqrt(v+eps))


        if stepchoice==0:
            x[:] = x - (step0/L) * sg
        elif stepchoice>0:
            sk = float(step0/((k+1)**stepchoice))
            x[:] = x - sk * sg

        nx = torch.norm(x) #Computing the norm to measure divergence


        if average:
            # If average, compute the average of the iterates
            xavg = k/(k+1) *xavg + x/(k+1)
            nmin = torch.norm(xavg-xtarget)
            obj = problem.obj_func(xavg)
        else:
            obj = problem.obj_func(x)
            nmin = torch.norm(x-xtarget)
        #########################################


        k += 1
        # Plot quantities of interest at the end of every epoch only
        if (k*nb) % n == 0:
            objvals.append(obj)
            normits.append(nmin)
            if verbose:
                print(' | '.join([("%d" % k).rjust(8),("%.2e" % obj).rjust(8),("%.2e" % nmin).rjust(8)]))

    # End of main loop
    #################

    # Plot quantities of interest for the last iterate (if needed)
    if (k*nb) % n > 0:
        objvals.append(obj)
        normits.append(nmin)
        if verbose:
            print(' | '.join([("%d" % k).rjust(8),("%.2e" % obj).rjust(8),("%.2e" % nmin).rjust(8)]))

    # Outputs
    if average:
        x_output = xavg.copy()
    else:
        x_output = x.clone().detach()

    return x_output, np.array(objvals), np.array(normits)


def proximal_stochastic_grad(problem, lbda, stepchoice=0, step0=1, n_epoch=None, n_iter=1000, nb=1, with_replace=False, verbose=True):
    """
    Performs proximal stochastic gradient descent optimization.

    Inputs:
        problem: Problem instance containing loss and objective functions
        lbda: Regularization parameter for the proximal step
        stepchoice: Step size selection strategy
            0: Constant step size of 1/L
            t > 0: Decreasing step size 1/(k+1)^t
        step0: Initial step size
        n_epoch: Number of epochs (overrides n_iter if provided)
        n_iter: Number of iterations
        nb: Batch size for stochastic updates
        with_replace: If True, samples components with replacement
        verbose: If True, prints progress at each iteration

    Outputs:
        x_output: Final iterate of the method
        objvals: History of objective values
        normits: History of iterate norms
    """
    objvals = []  # Objective value history
    normits = []  # Norm of iterates history
    L = problem.lipgrad()  # Lipschitz constant
    n = problem.n  # Number of components
    x = problem.x.clone().detach()  # Initial iterate

    obj = problem.obj_func(x).detach()  # Initial objective value
    objvals.append(obj)
    nx = torch.norm(x)  # Norm of the iterate
    normits.append(nx)

    if verbose:
        print("Proximal Stochastic Gradient, batch size =", nb, "/", n)
        print(' | '.join([name.center(8) for name in ["iter", "fval", "normit"]]))
        print(' | '.join([("%d" % 0).rjust(8), ("%.2e" % obj).rjust(8), ("%.2e" % nx).rjust(8)]))

    if n_epoch is not None:
        n_iter = n_epoch * (n // nb)

    for k in range(n_iter):
        # Draw batch indices
        ik = np.random.choice(n, nb, replace=with_replace)

        # Compute batch gradient
        sg = torch.zeros_like(x)
        for j in range(nb):
            sg += problem.grad_explicit(x, ik[j])  # Use explicit gradient for stochastic updates
        sg /= nb  # Average gradient over the batch

        # Update step size
        if stepchoice == 0:
            step = step0 / L
        elif stepchoice > 0:
            step = float(step0 / ((k + 1) ** stepchoice))

        # Perform proximal gradient step
        x -= step * sg
        for i in range(len(x)):
            threshold = step * lbda
            if x[i] < -threshold:
                x[i] += threshold
            elif x[i] > threshold:
                x[i] -= threshold
            else:
                x[i] = 0

        # Update norm and objective value
        nx = torch.norm(x)
        obj = problem.obj_func(x).detach()
        objvals.append(obj)
        normits.append(nx)

        if verbose and (k + 1) % (n // nb) == 0:
            print(' | '.join([("%d" % (k + 1)).rjust(8), ("%.2e" % obj).rjust(8), ("%.2e" % nx).rjust(8)]))

    return x.clone().detach(), np.array(objvals), np.array(normits)


def stochastic_BFGS(problem, stepchoice=0, step0=1, n_epoch=30, nb=1, with_replace=False, verbose=True):
    N = int(problem.n / nb)
    x = problem.x.clone().detach()
    H = torch.eye(problem.d)
    L = problem.lipgrad()
    objvals = []

    for epoch in range(n_epoch):
        for i in range(N):
            ik = torch.randint(0, problem.n, (nb,), generator=None if not with_replace else torch.Generator())
            sg = torch.zeros(problem.d)

            for j in range(nb):
                gi = compute_grad_i(problem.obj_func_i, x, ik[j])
                sg += gi
            sg /= nb

            if stepchoice == 0:
                step_size = step0 / L
            else:
                step_size = step0 / ((epoch * N + i + 1) ** stepchoice)

            step_size = max(step_size, 1e-8)
            delta_x = -step_size * H @ sg
            x_new = x + delta_x.flatten()

            s = delta_x
            vg = torch.zeros(problem.d)
            for j in range(nb):
                gi_next = compute_grad_i(problem.obj_func_i, x_new, ik[j])
                vg += gi_next
            vg /= nb

            v = vg - sg
            s_T_v = s.T @ v
            if s_T_v > 1e-5:
                term1 = (v @ s.T) / s_T_v
                H = (torch.eye(problem.d) - term1) @ H @ (torch.eye(problem.d) - term1.T) + (s @ s.T) / s_T_v
            else:
                H = H  # No update if the condition isn't met

            x = x_new

        fval = problem.obj_func(x)
        if not torch.isfinite(fval):
            raise ValueError("Objective function returned non-finite value.")
        objvals.append(fval.item())

        if verbose:
            print(f"Epoch {epoch + 1}/{n_epoch}, Objective value: {fval:.6f}")

    return objvals, x
