using AdvancedHMC
n_samples, n_adapts, target = 10_000, 2_000, 0.8 # set up sampling parameters
q0 = randn(D)   # draw a random starting point
### Building up NUTS
metric = DiagEuclideanMetric(D) # diagonal Euclidean metric space
h = Hamiltonian(metric, logdensity_f, grad_f)   # Hamiltonian on the target distribution
eps_init = find_good_eps(h, q0) # initial step size
int = Leapfrog(eps_init)    # leapfrog integrator
traj = NUTS{Multinomial,GeneralisedNoUTurn}(int)    # multi. sampling with gen. no U-turn   
adaptor = StanHMCAdaptor(   # Stan's windowed adaptor
    n_adapts, Preconditioner(metric), NesterovDualAveraging(target, eps_init)
)
samples, stats = sample(h, traj, q0, n_samples, adaptor, n_adapts)  # draw samples