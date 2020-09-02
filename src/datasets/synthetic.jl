"""
  genGaussian(mean_in, cov_in, class_label, n)

Draw from a gaussian distribution
# Arguments
- `mean_in` : means
- `cov_in` : covariances
- `class_label` : class_label
- `n` : number of samples to draw
"""
function genGaussian(mean_in, cov_in, class_label, n)
  nv = Distributions.MvNormal(mean_in, cov_in)
  X = rand(nv, n)
  y = ones(n) .* class_label
  return nv, transpose(X), y
end

"""
  genZafarData(n = 10000, d = pi/4)

Generate synthetic data from Zafar et al., 2017 Fairness Constraints: Mechanisms for Fair Classification
# Arguments
- `n=10000` : number of samples
- `d=pi/4` : discrimination factor
# Returns
- `X` : DataFrame containing features and protected attribute z
- `y` : Binary Target variable
"""
function genZafarData(n = 10000, d = pi/4)
    mu1, sigma1 = [2., 2.],  [[5., 1.] [1., 5.]]
    mu2, sigma2 = [-2.,-2.], [[10., 1.]  [1., 3.]]
    nv1, X1, y1 = genGaussian(mu1, sigma1, 1, Int64(floor(n/2))) # positive class
    nv2, X2, y2 = genGaussian(mu2, sigma2, -1, Int64(floor(n/2))) # negative class
    perm = shuffle(1:(2*Int64(floor(n/2))))
    X = vcat(X1, X2)[perm,:]
    y = vcat(y1, y2)[perm]
    rotation_mult = [[cos(d), -sin(d)] [sin(d), cos(d)]]
    X_aux = X * rotation_mult
    """ Generate the sensitive feature here """
    x_control = [] # this array holds the sensitive feature value
    for i in 1:(2*Int64(floor(n/2)))
        x = X_aux[i,:]
        # probability for each cluster that the point belongs to it
        p1 = Distributions.pdf(nv1, x)
        p2 = Distributions.pdf(nv2, x)
        # normalize the probabilities from 0 to 1
        s = p1+p2
        p1 = p1/s
        p2 = p2/s
        if rand(1)[1] < p1 # the first cluster is the positive class
            append!(x_control, 1.0)
        else
            append!(x_control, 0.0)
        end
    end
    X = DataFrame(X)
    X.z = x_control
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
- `y` : Binary Target variable
"""
function genSubgroupData(n = 10000, setting = "B00")
    xdists = [
      Distributions.Normal(0.,1.), # x1 is N(0,1)
      Distributions.MvNormal([0., 0.], [[1, 0.5] [0.5, 1]]), # x2 and x3 are N(0,1) with cor 0.5
      Distributions.Exponential(1.), # x4 is exp(1)
      Distributions.Bernoulli(0.5), # x5 is Ber(0.5)
      Distributions.Categorical(repeat([.1], 10)), # x6 is Multinom. w. equal
      Distributions.MvNormal([0., 0., 0., 0.], [[1, 0.5, 0.5, 0.5] [0.5, 1, 0.5, 0.5] [0.5, 0.5, 1, 0.5] [0.5, 0.5, 0.5, 1]]
      ) # x7 to 10 are N(0,1) with cor 0.5
    ]
    z = rand(Distributions.Bernoulli(0.5), n)
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
    X = DataFrame(X)
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
  return ifelse.(yprob .> rand(1)[1], 1, 0)
end
