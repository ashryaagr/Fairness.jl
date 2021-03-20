"""
  genGaussian(mean_in, cov_in, class_label, n)

Draw from a gaussian distribution
# Arguments
- `mean_in` : means
- `cov_in` : covariances
- `class_label` : class_label
- `n` : number of samples to draw
"""
function genGaussian(mean_in::Array, cov_in::Array, class_label::Int64, n::Int64)
  nv = Distributions.MvNormal(mean_in, cov_in)
  X = rand(nv, n)
  y = ones(Int64, n) .* class_label
  return nv, transpose(X), y
end

"""
  genZafarData(n = 10000; d = pi/4)

Generate synthetic data from Zafar et al., 2017 Fairness Constraints: Mechanisms for Fair Classification.

# Arguments
- `n=10000` : number of samples
- `d=pi/4` : discrimination factor

# Returns
- `X` : DataFrame containing features and protected attribute z {"A", "B"} where
        z="B" is the protected group.
- `y` : Binary Target variable {-1, 1}
"""
function genZafarData(n = 10000; d = pi/4)
    mu1, sigma1 = [2., 2.],  [[5., 1.] [1., 5.]]
    mu2, sigma2 = [-2.,-2.], [[10., 1.]  [1., 3.]]
    npos = Int64(floor(n/2))
    nv1, X1, y1 = genGaussian(mu1, sigma1,  1, npos)     # positive class
    nv2, X2, y2 = genGaussian(mu2, sigma2, -1, n - npos) # negative class
    perm = shuffle(1:n)
    X = vcat(X1, X2)[perm,:]
    y = vcat(y1, y2)[perm]
    rotation_mult = [[cos(d), -sin(d)] [sin(d), cos(d)]]
    X_aux = X * rotation_mult
    """ Generate the sensitive feature here """
    x_control = Array{String, 1}(undef, n) # this array holds the sensitive feature value
    for i in 1:n
        x = X_aux[i,:]
        # probability for each cluster that the point belongs to it
        p1 = Distributions.pdf(nv1, x)
        p2 = Distributions.pdf(nv2, x)
        # normalize the probabilities from 0 to 1
        s = p1+p2
        p1 = p1/s # p2 = p2/s
        if rand(1)[1] < p1
            x_control[i] = "A" # majority class
        else
            x_control[i] = "B" # protected class
        end
    end
    X = DataFrame(X, :auto)
    X.z = categorical(x_control)
    coerce!(X, :z => Multiclass)
    y = categorical(y)
    return X, y
end



"""
Generate synthetic data from Zafar et al., 2017 Fairness Beyond Disparate Treatment & Disparate Impact
# Arguments
- `n=10000` : number of samples
# Returns
- `X` : DataFrame containing features and protected attribute z
- `y` : Binary Target variable
"""
function genZafarData2(n = 10000)

    y = rand(Distributions.Bernoulli(0.5), n)
    z = rand(Distributions.Bernoulli(0.5), n)
    # Depending on z and y, we draw from one of the 4 distributions in M.
    M = [
      [Distributions.MvNormal([1., 1.],    [[3., 1.] [1., 3.]]), Distributions.MvNormal([2., 2.], [[3., 1.] [1., 3.]])],
      [Distributions.MvNormal([-2., -2.],  [[3., 1.] [1., 3.]]), Distributions.MvNormal([2., 2.], [[3., 1.] [1., 3.]])]
      ]

    # Iterate over z and y in parallel, drawing from the appropritate Distribution.
    for i in 1:n
      if i == 1
        X = rand(M[z[i]+1][y[i]+1], 1)
      else
        X = hcat(X, rand(M[z[i]+1][y[i]+1], 1))
      end
    end

    perm = shuffle(1:n)
    X = DataFrame(transpose(X)[perm,:], :auto)
    y = y[perm]
    X.z = [x == 0 ? "A" : "B" for x in z[perm]]
    coerce!(X, :z => Multiclass)
    y = categorical(y)
    return X, y
end



"""
  genSubgroupData(n=10000, setting="B00")

Generate synthetic data from Loh et al., 2019 : Subgroup identification for precision medicine: A comparative review of 13 methods
# Arguments
- `n=10000` : number of samples
- `setting="B00"` : Simulation data setting: one of "B00", ..., "B02", "B1", ... , "B8"
For "B00", ..., "B02" there is no "bias" in the data, i.e. group membership has no effect on y.
whereas for  "B1", ... , "B8", there is a direct effect of group membership z on y, usually mediated by one or more features.
# Returns
- `X` : DataFrame containing features and protected attribute z
- `y` : Binary target variable
"""
function genSubgroupData(n = 10000; setting = "B00")
    xdists = [
      Distributions.Normal(0.,1.), # x1 is N(0,1)
      Distributions.MvNormal([0., 0.], [[1, 0.5] [0.5, 1]]), # x2 and x3 are N(0,1) with cor 0.5
      Distributions.Exponential(1.), # x4 is exp(1)
      Distributions.Bernoulli(0.5), # x5 is Ber(0.5)
      Distributions.Categorical(repeat([.1], 10)), # x6 is Multinom. w. equal
      Distributions.MvNormal([0., 0., 0., 0.], [[1, 0.5, 0.5, 0.5] [0.5, 1, 0.5, 0.5] [0.5, 0.5, 1, 0.5] [0.5, 0.5, 0.5, 1]]
      ) # x7 to 10 are N(0,1) with cor 0.5
    ]
    z = [x == 0 ? "A" : "B" for x in rand(Distributions.Bernoulli(0.5), n)]
    X = []
    for d in xdists
      if length(X) > 0
        xn = rand(d, n)
        if size(xn)[1] != n
          xn = transpose(xn)
        end
        X = hcat(X, xn)
      else
        X = rand(d, n)
      end
    end
    y = logit_fun(X, z, setting)
    X = DataFrame(X, :auto)
    X.z = z
    coerce!(X, :z => Multiclass)
    y = categorical(y)
    return X, y
end

# Logistic Link function
function log_link(x)
  return exp(x)/(1+exp(x))
end

"""
  logit_fun(X, z, setting)

Compute y from X and z according to a setting provided in Loh et al., 2019: Subgroup identification for precision medicine: A comparative review of 13 methods
# Arguments
- `X` : matrix of features
- `z` : vector of group assignments
- `setting` : Simulation data setting: one of "B00", ..., "B02", "B1", ... , "B8"
"""
function logit_fun(X, z, setting)
  z = z == "A" ? 1 : 0
  if setting == "B00"
    logit = repeat([0.], length(z))
  elseif setting == "B01"
    logit = 0.5*(X[:,1] + X[:,2])
  elseif setting == "B02"
    logit = 0.5*(X[:,1] + X[:,2]) .^ 2
  elseif setting == "B1"
    logit = 0.5*(X[:,1] + X[:,2] - X[:,5]) + 2. * z .* map(x -> ifelse(isodd(Int64(x)), 1., 0.), X[:,6])
  elseif setting == "B2"
    logit = 0.5*X[:,2] + 2. * z .* map(x -> ifelse(x > 0, 1., 0.), X[:,1])
  elseif setting == "B3"
    logit = 0.3*(X[:,1] + X[:,2]) + 2. * z .* map(x -> ifelse(x > 0, 1., 0.), X[:,1])
  elseif setting == "B4"
    logit = 0.2*(X[:,2] + X[:,3] .- 2.) + 2. * z .* X[:,4]
  elseif setting == "B5"
    logit = 0.2*(X[:,1] + X[:,2] .- 3) + 2. * z .* map(x -> ifelse(x < 0, 1., 0.), X[:,1]) .* map(x -> ifelse(isodd(Int64(x)), 1., 0.), X[:,6])
  elseif setting == "B6"
    logit = 0.5*(X[:,2] .- 1.) + 2. * z .* map(x -> ifelse(abs(x) < 0.8, 1., 0.), X[:,1])
  elseif setting == "B7"
    logit = 0.2*(X[:,2] + X[:,2] .^ 2 .- 6.)  + 2. * z .* map(x -> ifelse(x < 0, 1., 0.), X[:,1])
  elseif setting == "B8"
    logit = 0.5 * X[:,2] + 2. * z .* X[:,5]
  end
  yprob = log_link.(logit)
  # Compute a random y with probability yprob
  u = rand(length(yprob))
  return yprob .> u
end

"""
  genBiasedSampleData(n=10000, sampling_bias=0.8)

Generate synthetic data: Biased sample

# Arguments
- `n=10000` : number of samples
- `sampling_bias=0.8` : Percentage of data belonging to majority group.

The idea behind this simulation is that algorithms might fit the process in the
majority group while disregarding the process in the minority group.

Two different processes for d1 and d2:
d1: logit(y) = 0.5*(    X1 + X2 + 0.3*X4) + 2*(I(X3 > 0))
d2: logit(y) = 0.5*(0.3*X1 + X2 +     X4) + 2*(I(X3 > 0.2))

# Returns
- `X` : DataFrame containing features and protected attribute z
- `y` : Binary Target variable
"""
function genBiasedSampleData(n = 10000; sampling_bias = 0.8)

  d1 = Distributions.MvNormal([0., 0., 0., 0.], [[1, 0.3, 0.3, 0.3] [0.3, 1, 0.3, 0.3] [0.3, 0.3, 1, 0.3] [0.3, 0.3, 0.3, 1]])
  d2 = Distributions.MvNormal([0.2, 0.2, 0.2, 0.2], [[1, 0.5, 0.5, 0.5] [0.5, 1, 0.5, 0.5] [0.5, 0.5, 1, 0.5] [0.5, 0.5, 0.5, 1]])

  s1 = Int64(floor(n * sampling_bias))

  X1 = transpose(rand(d1, s1))
  X2 = transpose(rand(d2, n-s1))

  perm = shuffle(1:n)
  yprob = vcat(
    log_link.(0.5 * (       X1[:,1] + X1[:,2] + 0.3 * X1[:,4]) + map(x -> ifelse(x > 0, 2., 0.), X1[:,3])),
    log_link.(0.5 * (0.3 .* X2[:,2] + X2[:,2] +       X2[:,4]) + map(x -> ifelse(x > .2, 2., 0.), X2[:,3]))
  )[perm]
  X = vcat(X1, X2)[perm,:]
  z = vcat(repeat(["A"], s1), repeat(["B"], n-s1))[perm]

  X = DataFrame(X, :auto)
  X.z = z
  coerce!(X, :z => Multiclass)
  u = rand(n)
  y = categorical(ifelse.(yprob .> u, 1, 0))
  return X, y
end

"""
returns R, S, A, G, L, F
see https://github.com/apedawi-cs/Causal-inference-discussion/blob/master/law_school.ipynb
"""
function law_school(nb_obs, R_pct=0.75, S_pct=0.6)
  function simulate_exogenous_vars(nb_obs, R_pct=0.5, S_pct=0.5)
    R = rand(Distributions.Uniform(0, 1), nb_obs, 1) .< R_pct
    S = rand(Distributions.Uniform(0, 1), nb_obs, 1) .< S_pct
    A = randn(nb_obs, 1)
    return R, S, A
  end
  function simulate_endogenous_vars(A, R, S)
    nb_obs = length(A)
    G = A + 2.1 * R + 3.3 * S + 0.5 * randn(nb_obs, 1)
    L = A + 5.8 * R + 0.7 * S + 0.1 * randn(nb_obs, 1)
    F = A + 2.3 * R + 1.0 * S + 0.3 * randn(nb_obs, 1)
    return G, L, F
  end
  R, S, A = simulate_exogenous_vars(nb_obs, R_pct, S_pct)
  G, L, F = simulate_endogenous_vars(A, R, S)
  R, S, A, G, L, F
end
