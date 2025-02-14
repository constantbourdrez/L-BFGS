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



def adv_stoch_grad(problem,xtarget,stepchoice=0,step0=1, n_epoch = None, n_iter=1000,nb=1,average=0,scaling=0,with_replace=False,verbose=False):
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
    compt_epoch = 1
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
        if (k*nb) > n*compt_epoch:
            compt_epoch += 1
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

def stochastic_BFGS(problem, xmin, stepchoice=0, step0=1, n_epoch=30, nb=1, with_replace=False, verbose=True, proximal = False, lbda = None):
    """
    Stochastic BFGS implementation with formulas adapted from standard BFGS.
    Parameters:
        problem (object): An optimization problem instance
        xmin (torch.Tensor): Known optimal solution (for convergence monitoring).
        stepchoice (int, optional): Step size adaptation strategy.
                                    - 0: Uses a step size proportional to the inverse Lipschitz constant.
                                    - Other values: Uses a decreasing step size of the form `step0 / (iteration_count ^ stepchoice)`.
                                    Default is 0.
        step0 (float, optional): Initial step size scaling factor. Default is 1.
        n_epoch (int, optional): Number of training epochs. Default is 30.
        nb (int, optional): Mini-batch size. Default is 1.
        with_replace (bool, optional): Whether to sample with replacement. Default is False.
        verbose (bool, optional): If True, prints optimization progress. Default is True.
        proximal (bool, optional): If True, applies proximal updates (e.g., for regularization). Default is False.
        lbda (float, optional): Regularization parameter for proximal updates. Required if `proximal=True`.

    Returns:
        torch.Tensor: Optimized variable `x`.
        numpy.ndarray: Objective function values over iterations.
        numpy.ndarray: Distance of iterates from `xmin` over iterations.

    """
    N = int(problem.n / nb)  # Number of mini-batches per epoch
    x = problem.x.clone().detach().double()  # Initialize x and ensure dtype is double
    H = torch.eye(problem.d, dtype=torch.float64)  # Initialize Hessian approximation as identity matrix
    objvals = []  # Store objective function values
    # iterates distance to the minimum history
    normits = []

    for epoch in range(n_epoch):
        for i in range(N):
            # Sample mini-batch indices
            ik = torch.randint(0, problem.n, (nb,), generator=None if not with_replace else torch.Generator())

            # Compute stochastic gradient for the mini-batch
            sg = torch.zeros(problem.d, dtype=torch.float64)
            for j in range(nb):
                gi = compute_grad_i(problem.obj_func_i, x, ik[j])
                sg += gi
            sg /= nb

            # Step size adjustment
            if stepchoice == 0:
                step_size = step0 / problem.lipgrad()  # Lipschitz constant
            else:
                step_size = step0 / ((epoch * N + i + 1) ** stepchoice)
            step_size = max(step_size, 1e-8)

            # Compute search direction and update x
            p = -step_size * H @ sg
            x_new = x + p

            # Compute s and y
            s = (x_new - x).view(-1, 1)
            vg = torch.zeros(problem.d, dtype=torch.float64)
            for j in range(nb):
                gi_next = compute_grad_i(problem.obj_func_i, x, ik[j])
                vg += gi_next
            vg /= nb
            y = (vg - sg).view(-1, 1)


            sT = s.T
            yT = y.T
            yT_s = yT @ s
            if yT_s > 1e-8:
                rho = 1.0 / yT_s
                rho2 = rho**2
                I = torch.eye(problem.d, dtype=torch.float64)

                # Hessian update
                H_y = H @ y
                H_new = (
                    H
                    - rho * (H_y @ sT + s @ (yT @ H))
                    + rho2 * ((s @ (yT @ H_y)) @ sT)
                    + rho * (s @ sT)
                )
                H = H_new

            x = x_new
            if proximal:
                for i in range(len(x)):
                    threshold = step_size * lbda
                    if x[i] < -threshold:
                        x[i] += threshold
                    elif x[i] > threshold:
                        x[i] -= threshold
                    else:
                        x[i] = 0



        fval = problem.obj_func(x)
        if not torch.isfinite(fval):
            raise ValueError("Objective function returned non-finite value.")
        objvals.append(fval.item())
        normits.append(torch.norm(x - xmin).item())

        if verbose:
            print(f"Epoch {epoch + 1}/{n_epoch}, Objective value: {fval:.6f}")

    return x, np.array(objvals), np.array(normits)

def stochastic_LBFGS(problem, xmin, memory_size=5, c=0.0001, theta=0.5, n_epoch=30, nb=32, verbose=True, proximal = False, lbda = None):
    """
    Stochastic L-BFGS optimization algorithm.

    This function implements a stochastic variant of the Limited-memory BFGS (L-BFGS)
    algorithm, which approximates the inverse Hessian using a history of recent updates.


    Parameters:
        problem (object): An optimization problem instance with methods `obj_func_i`
                          (per-sample objective) and `compute_grad_i`.
        xmin (torch.Tensor): Known optimal solution (for convergence monitoring).
        memory_size (int, optional): Number of past updates to store in memory (controls
                                     Hessian approximation quality). Default is 5.
        c (float, optional): Sufficient decrease parameter for backtracking line search. Default is 0.0001.
        theta (float, optional): Reduction factor for step size in backtracking. Default is 0.5.
        n_epoch (int, optional): Number of training epochs. Default is 30.
        nb (int, optional): Mini-batch size. Default is 32.
        verbose (bool, optional): If True, prints optimization progress. Default is True.
        proximal (bool, optional): If True, applies proximal updates (e.g., for regularization). Default is False.
        lbda (float, optional): Regularization parameter for proximal updates. Required if `proximal=True`.

    Returns:
        torch.Tensor: Optimized variable `x`.
        numpy.ndarray: Objective function values over iterations.
        numpy.ndarray: Distance of iterates from `xmin` over iterations.
    """
    N = int(problem.n / nb)  # Number of mini-batches per epoch
    x = problem.x.clone().detach().double()  # Initialize x and ensure dtype is double
    objvals = []  # Store objective function values
    normits = []  # Store iterates distance to the minimum history
    sk_memory = []
    yk_memory = []
    rho_memory = []

    for epoch in range(n_epoch):
        for j in range(N):
            # Sample mini-batch indices
            ik = torch.randint(0, problem.n, (nb,), dtype=torch.long)

            # Compute stochastic gradient for the mini-batch
            sg = torch.zeros(problem.d, dtype=torch.float64)
            for j in range(nb):
                gi = compute_grad_i(problem.obj_func_i, x, ik[j])
                sg += gi
            sg /= nb

            # L-BFGS Two-loop recursion
            q = sg.clone()
            alphas = []

            for i in reversed(range(len(sk_memory))):
                rho = rho_memory[i]
                alpha = rho * torch.dot(sk_memory[i], q)
                alphas.append(alpha)
                q -= alpha * yk_memory[i]

            # Approximate Hessian initialization (scaling)
            if len(sk_memory) > 0:
                gamma = torch.dot(sk_memory[-1], yk_memory[-1]) / torch.dot(yk_memory[-1], yk_memory[-1])
                q *= gamma

            # Second loop to compute search direction
            for i in range(len(sk_memory)):
                rho = rho_memory[i]
                beta = rho * torch.dot(yk_memory[i], q)
                q += sk_memory[i] * (alphas[len(sk_memory) - 1 - i] - beta)

            direction = -q

            # Perform line search (simple backtracking)
            step_size = 1.0
            while problem.obj_func(x + step_size * direction) > problem.obj_func(x) + c * step_size * torch.dot(sg, direction):
                step_size *= theta  #Got rid of the stepchoice since it wasn't working here

            x_new = x + step_size * direction
            sk = x_new - x

            # Compute new stochastic gradient
            vg = torch.zeros(problem.d, dtype=torch.float64)
            for j in range(nb):
                gi_next = compute_grad_i(problem.obj_func_i, x_new, ik[j])
                vg += gi_next
            vg /= nb
            yk = vg - sg

            # Update memory
            if len(sk_memory) >= memory_size:
                sk_memory.pop(0)
                yk_memory.pop(0)
                rho_memory.pop(0)
            sk_memory.append(sk)
            yk_memory.append(yk)
            rho_memory.append(1.0 / torch.dot(yk, sk))

            x = x_new
            if proximal:
                for i in range(len(x)):
                    threshold = step_size * lbda
                    if x[i] < -threshold:
                        x[i] += threshold
                    elif x[i] > threshold:
                        x[i] -= threshold
                    else:
                        x[i] = 0

            if verbose:
                print(f"Iteration {batch + epoch * N}: objective = {problem.obj_func(x).item()}")

        f = problem.obj_func(x)
        objvals.append(f.item())
        normits.append(torch.norm(x - xmin).item())

    return x, np.array(objvals), np.array(normits)
