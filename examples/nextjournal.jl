# # [Fairness.jl](https://github.com/ashryaagr/Fairness.jl) - Fairness Toolkit in Julia
#
# Fairness.jl is a new bias audit and mitigation toolkit in Julia designed with the aim to solve practical problems faced by practitioners with existing fairness toolkits.
#
# This notebook shall present you with an introduction to **[Fairness.jl](https://github.com/ashryaagr/Fairness.jl)**, its power, and uniqueness by the means of a Real-Life example of [COMPAS Dataset](https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb).
#
# But before visiting we begin the introduction, let us be clear why we need this package.
#
# ## Why should  I care about fairness, ethics, etc. ?
#
# Machine Learning is involved in a lot of crucial decision support tools. The use of these tools ranges from granting parole, shortlisting job applications to accepting credit applications. There have been a lot of political and policy developments during the past year that have pointed out the transparency issues and data bias in these Machine Learning tools. Thus it has become crucial for the Machine learning community to think about fairness and bias. Eliminating bias is not as easy as it might seem at first glance. This toolkit helps you in easily auditing and minimizing the bias through a collection of various fairness metrics and algorithms.
#
# # Example
#
# In this example, we will use the COMPAS Dataset to predict whether a criminal defendant will recidivate(re-offend). Neural Network Classifier is used for classification. It is wrapped with the Reweighing Algorithm to preprocess the data. This wrapped Model is then wrapped with Equalized Odds Algorithm for postprocessing of predictions.
#
# ## Downloading Required Packages
#
# To install the package, you have to install
#
# * Fairness.jl package
# * [MLJ : Machine Learning Toolkit](https://alan-turing-institute.github.io/MLJ.jl/dev/)
# * PrettyPrinting : To get pretty output.`pprint`is its only function used by us.

using Pkg
Pkg.activate("my_environment", shared=true)
Pkg.add("Fairness")
Pkg.add("MLJ") # Toolkit for Machine Learning
Pkg.add("PrettyPrinting") # For readibility of outputs

# Now we import the required packages. Note that this will take 5 minutes on the first run. This is the case with all Julia packages. Julia pre-compiles the code to make it more efficient. This is a one-time thing. 2nd run onwards, everything will be fast!!
#

using Fairness
using MLJ
using PrettyPrinting




#
# ## Load the [COMPAS Dataset](https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb) using the macro by Fairness.jl
#
# This dataset has 8 features and 6907 rows. The protected attribute here is `race`. Using the 8 features, it predicts whether a criminal defendant will recidivate(re-offend).
#

data = @load_compas
X, y = data
data |> pprint

#
# ### Multi-valued Protected Attribute
#
# Notice in the output of the previous cell that the column `race` has 6 different possible values "Native American", "African-American", *"Caucasian"*, "Hispanic", "Asian" and "Other".
#
# We support Multi-valued Protected attributes in both fairness algorithms and metrics. *The fairness algorithms by Researchers have been improved to generalize for multiple values of a protected attribute.*
#
# ## Load Neural Network Classifier
#
# We will use [MLJFlux](https://github.com/alan-turing-institute/MLJFlux.jl) to load Neural Network classifier. MLJFlux is an interface of Flux with MLJ. You don't need to explicitly import MLJFlux. MLJ does all that for you!!
#
# We use the `@load` macro to load Neural Network Classifier into main. Then using the Neural Network, we use `@pipeline` to add Continuous Encoder to the Neural Network Encoder. Continuous Encoder converts categorical strings to continuous values that support a much wider range(\~50) of models!!
#
Pkg.add("MLJFlux")
@load NeuralNetworkClassifier
model = @pipeline ContinuousEncoder NeuralNetworkClassifier

#
# ## Fairness Algorithm Wrappers
#
# We first wrap the Neural Network Classifier with Reweighing Algorithm. This wrapped Model is again wrapped with LinProg Postprocessing Algorithm.
#
# *Notice how usage of wrappers allows provides us with composability and enables you to apply unlimited algorithms on a single classifier.*
#
wrappedModel = ReweighingSamplingWrapper(classifier=model, grp=:race)
wrappedModel2 = LinProgWrapper(classifier=wrappedModel, grp=:race,
                                measure=false_positive_rate)

# ## Automatic evaluation using MLJ.evaluate
#
# Using the evaluate function from MLJ, you only need to pass your model, data and concerned metrics. MLJ handles the rest of the work internally. Note that you need to wrap the metrics to specify the protected attribute.
#
evaluate(wrappedModel2,
  X, y,
  measures=[
    Disparity(false_positive_rate, refGrp="Caucasian", grp=:race),
    MetricWrapper(accuracy, grp=:race)]) |> pprint

# # Finer Control (Advanced)
#
# You can get greater control than what is provided by `evaluate` function.
#
# * First, we need to get the train and test indices. This will be provided by the partition function.
# * `machine` is used to package the dataset and the wrapped Model (reused from before)
# * The machine fitted on the training rows.
#

train, test = partition(eachindex(y), 0.7, shuffle=true)
mach = machine(wrappedModel2, X, y)
fit!(mach, rows=train)

#
# Now we use `predict` function on the `machine` on the rows specified by `test`
#

ŷ = predict(mach, rows=test)
ŷ |> pprint

#
# ## Auditing Bias
#
# We use the concept of Fairness Tensors to avoid redundant calculations. Refer <https://www.ashrya.in/Fairness.jl/dev/fairtensor/> to learn more about Fairness Tensors.
#
# We pass predictions, ground-truth and protected values to the `fair_tensor` function.
#

ft = fair_tensor(ŷ, y[test], X[test, :race])
#
# ### [Disparity Calculation](https://www.ashrya.in/Fairness.jl/dev/measures/#Fairness.disparity)
#
# Disparity can be calculated by passing the following to the disparity `function` :
#
# * An array of fairness metrics from the ones [listed in README](https://www.ashrya.in/Fairness.jl/dev/fairtensor/)
# * Fairness tensor that we calculated in the previous step
# * Reference Group
# * `func` : disparity value for a metric M, group A and reference group B is `func(M(A), M(B))` . The default value for func is division and hence is an optional argument
#

df_disparity = disparity(
  [accuracy, false_positive_rate],
  ft, refGrp="Caucasian",
  func=(x, y)->(x-y)/y
)

#
# The values above show that Asians and African-Americans have a higher percentage of False Positive Rate w.r.t. the reference group Caucasian. On the other hand, Native Americans have a lower percentage of False Positive Rate w.r.t Caucasian. But these disparity values are better than the case if the Neural Network Classifier was directly used.
#
# ## [Parity Calculation](https://www.ashrya.in/Fairness.jl/dev/measures/#Fairness.parity)
#
# To calculate parity, we need to pass following to the `parity` function :
#
# * DataFrame output from the disparity function in the previous step
# * Custom Function to calculate parity based on disparity values.
#
# Scroll the output to the right to see the column for parity values.
#

parity(df_disparity,
  func= (x) -> abs(x)<0.4
)


#
# The above parity outputs show that parity constraints for False Positive Rate are satisfied only by the groups: Other and Caucasian.
#
# # Visualizing improvement by Fairness Algorithm
#
# Now we will use [VegaLite](https://www.queryverse.org/VegaLite.jl/stable/) to visualize the improvement in fairness metrics due to the fairness algorithms added in the form of wrappers. We shall also visualize the drop in accuracy due to the trade-off between accuracy and fairness.
#
# Note that **`wrappedModel2`** is the ML model we previously wrapped with Reweighing algorithm and LinProg Algorithm.
#
# Summary of what following code does :
#
# * Evaluate metric values using MLJ.evaluate for both: The Wrapped Model and the original model
# * Collect metric values  from the result of evaluate function
# * Create a DataFrame using the collected values that will later be used with VegaLite to plot the graphs
#
Pkg.add("VegaLite")
Pkg.add("DataFrames")
using VegaLite
using DataFrames

result = evaluate(wrappedModel2,
    X, y,
    measures=[
    Disparity(false_positive_rate, refGrp="Caucasian", grp=:race),
    MetricWrapper(accuracy, grp=:race)])

result_1 = evaluate(model,
    X, y,
    measures=[
    Disparity(false_positive_rate, refGrp="Caucasian", grp=:race),
    MetricWrapper(accuracy, grp=:race)])

n_grps = length(levels(X[!, :race]))
dispVals = collect(values(result.measurement[1]))
dispVals_1 = collect(values(result_1.measurement[1]))
accVals = collect(values(result.measurement[2]))
accVals_1 = collect(values(result_1.measurement[2]))

df = DataFrame(
  disparity=vcat(dispVals, dispVals_1),
  accuracy=vcat(accVals, accVals_1),
  algo=vcat(repeat(["Wrapped Model"],n_grps+1), repeat(["ML Model"],n_grps+1)),
  grp=repeat(collect(keys(result.measurement[1])), 2));

#
# ## Improvement in False Positive Rate Disparity Values
#

df |> @vlplot(
  :bar,
  column="grp:o",
  y={"disparity:q",axis={title="False Positive Rate Disparity"}},
  x={"algo:o", axis={title=""}},
  color={"algo:o"},
  spacing=20,
  config={
  view={stroke=:transparent},
  axis={domainWidth=1}
  }
)

#
#
# The above plot shows that there was a high bias against the group "African-American" in the NeuralNetworkClassifier(ML Model). The False Positive Rate Disparity value is greater than `2.0` for this group while its nearly to `1.0` for others. This means that a person belonging to the group "African-American" is twice as likely as other groups to be falsely predicted as a criminal who would re-offend!!
#
# But in the case of the wrapped model, the False Positive Rate disparity has been reduced for "African-American" to about `1.3` which is the same as most other groups.
#
# ## Accuracy Comparison
#
df |> @vlplot(
  :bar,
  column="grp:o",
  y={"accuracy:q",axis={title="Accuracy"}},
  x={"algo:o", axis={title=""}},
  color={"algo:o"},
  spacing=20,
  config={
  view={stroke=:transparent},
  axis={domainWidth=1}
  }
)
#
#
#
# The above plot shows that there is a drop in accuracy on using the wrapped model. This is a direct consequence of the fairness-accuracy tradeoff. So, we obtain a model that is fairer at the cost of accuracy.
#
# ## Fairness vs Accuracy Comparison across Algorithms
#
Pkg.add("Plots")
Pkg.add("PyPlot")
using Plots

function algorithm_comparison(algorithms, algo_names, X, y;
  refGrp, grp::Symbol=:class)
	grps = X[!, grp]
	categories = levels(grps)
	train, test = partition(eachindex(y), 0.7, shuffle=true)
	plot(title="Fairness vs Accuracy Comparison", seriestype=:scatter,
        xlabel="accuracy",
        ylabel="False Positive Rate Disparity refGrp="*refGrp,
        legend=:topleft, framestyle=:zerolines)
	for i in 1:length(algorithms)
		mach = machine(algorithms[i], X, y)
		fit!(mach, rows=train)
		ŷ = predict(mach, rows=test)
		if typeof(ŷ) <: MLJ.UnivariateFiniteArray
			ŷ = mode.(ŷ)
		end
		ft = fair_tensor(ŷ, y[test], X[test, grp])
		plot!([accuracy(ft)], [fpr(ft)/fpr(ft, grp=refGrp)],
      seriestype=:scatter, label=algo_names[i])
	end
	display(plot!())
end

algorithm_comparison([model, wrappedModel, wrappedModel2],
  ["NeuralNetworkClassifier", "Reweighing(Model)",
    "LinProg+Reweighing(Model)"], X, y,
    refGrp="Caucasian", grp=:race)

#
#
#
# # Concluding Remarks
#
# This toolkit has been designed to solve the numerous problems faced by both Policy Makers, Researchers, etc while using Fairness toolkits. Various innovative features of this package have been explicitly listed at <https://github.com/ashryaagr/Fairness.jl#what-fairnessjl-offers-over-its-alternatives>
#
# We are open to contributions. Feel free to open an issue on Github in case you want to contribute or have any confusion regarding the package. We would love to help you in getting started with this package.
#
# Finally, this work would have been impossible without the immense support, novel ideas, and efforts made by [Jiahao Chen](https://jiahao.github.io/), [Sebastian Vollmer](https://www.turing.ac.uk/people/researchers/sebastian-vollmer), and [Anthony Blaom](https://ablaom.github.io/).
#
# [Ashrya Agrawal](https://github.com/ashryaagr/) (ashryaagr@gmail.com)
#
# Link to Github Repository for Fairness.jl: <https://github.com/ashryaagr/Fairness.jl>
#
# Documentation: <https://www.ashrya.in/Fairness.jl/dev/>
#
# [nextjournal#output#96347bce-c38f-43fe-8426-d220e19780ed#result]:
# <https://nextjournal.com/data/QmdNHusG4ixmzbpU7FJ3WxxaRQVVA41o7MP5h5ibX7Ahh8?content-type=image/svg%2Bxml>
#
# [nextjournal#output#b0ab56a1-e5eb-4df3-b974-7a4319ac2b81#result]:
# <https://nextjournal.com/data/QmNUZfL9Kgu1hnujPKzbG8dXUJjPAf28pHcZwbpMJWTSpp?content-type=image/svg%2Bxml>
#
# [nextjournal#output#f69e3749-dccf-415c-be94-6d75b5f65639#result]:
# <https://nextjournal.com/data/QmRSVxCFZn2Hg7Lck2c7vEkcBaRcTqZi6hdUURUgdUwqZe?content-type=image/svg%2Bxml>
#
# <details id="com.nextjournal.article">
