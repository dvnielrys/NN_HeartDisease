### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ bd967d50-ba6d-11eb-0200-9962ce7b1a9c
begin
	using LinearAlgebra
	using Random
	using Statistics
	using Plots
	using CSV
	using DataFrames
	using ProgressMeter
	using StatsBase
	using LaTeXStrings
end

# ╔═╡ f55a1334-0365-4099-9bbe-3b3e0ddbfd43
md"
# Heart Disease predictor with an artificial neural network built from zero

###### By Servando Daniel López Reyes (18/06/2021)


The data was obtained from [kaggle.com](https://www.kaggle.com/ronitf/heart-disease-uci)

##### Context

This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The ''goal'' field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4.
"

# ╔═╡ 05f09de2-a658-47fa-be14-2d1aed589c44
data = CSV.read("heart.csv", DataFrame)

# ╔═╡ a209e430-000b-4f08-a2a0-db08e7f71363
md"Let´s do a normalization of the data in order to work with smaller values"

# ╔═╡ 60fb4aaa-fea4-41d1-b745-4e81ab616f04
df = DataFrame([data[:, i]/maximum(data[:, i]) for i in 1:14]);

# ╔═╡ 6bc223a4-665d-4d32-aa7a-0eaa56ce8145
md"### Neural Network Construction

As the first step we define de **activation functión** for the neurons of the network and the **cost function** that further we'll try to minimize with the **gradient descent algorithm**
"

# ╔═╡ 9f79f248-2129-4ed0-b312-fcfc13660a2d
#Sigmoid function
σ(x) = 1 / (1 + ℯ^(-x))

# ╔═╡ a03b9b05-45cf-4e08-9768-d2234bae49f9
#Sigmoid's derivative
dσ(x) = x * (1 - x)

# ╔═╡ 8fa3fc4f-8dd6-4772-bd88-51ee84b9da25
#Cost function
C(Yp, Yr) = -(Yr' * map(log, Yp)' + (ones(length(Yr)) - Yr)' * map(log, ones(length(Yp))' - Yp)')

# ╔═╡ a66ebfea-e2f4-4675-91ce-742071b18319
plot([σ], -5, 5, title="Activation function", lw=1.2,
    color="cadetblue", label="Sigmoid", legend=:topleft)

# ╔═╡ d3a26b86-9b6e-449e-a6de-b9232ab185e9
md"
Now, we create a function called `neural_layer`, it receive as parameters the number of neurons of the layer (`n_neur`) and the number of conections with the next layer (`n_conn`), and returns the **matrix** with the random initial values of a single layer and a vector with the **bias** values
"

# ╔═╡ 1985b05d-8ffc-41ff-8acb-ad7f37a0f7e8
function neural_layer(n_neur, n_conn)
    
    b = randn(n_neur, 1)
    W = randn(n_neur, n_conn) 
    
    return [W, b]
end

# ╔═╡ dde43202-2846-4121-afc1-8f7b7747f033
md"
As the next step we create the functión `create_nn` that build the **neural_network**, it receive as parameter a vector that contains the number of neurons of each layer. The function return the set of matrices and bias vectors that conform the network.
"

# ╔═╡ ea3d18e0-902f-46ab-8ec6-e68467460ec3
function create_nn(structure)
    
    nn = []
    
    for i in 1 : length(structure) - 1
        push!(nn, neural_layer(structure[i + 1], structure[i]))
    end
    
    return nn
end

# ╔═╡ 4452dac0-c423-46e1-bd97-2dc1039474d9
md"
Then, we make a function that will do the **forward propagation** of the inputs through the layers of the NN. The function receive the NN that we built before and the input $X$. We need a function that do the **backpropagation** as well.
"

# ╔═╡ a452c9da-54b5-4252-a104-5ec36799bb02
function forwardprop(nn, X)
    
    out = []
    push!(out, (nothing, X'))
    
    for i in 1:length(nn)
        Z = nn[i][1] * out[end][2] .+ nn[i][2]
        a = map(σ, Z)
        push!(out, (Z, a))
    end
    
    return out
end

# ╔═╡ cf7b9b26-81fc-4d90-88f4-939b213e5a80
function backprop(nn, X, Y)
    
    out = forwardprop(nn, X)
    δ = []
        
    for l in reverse(1:length(nn)) # We start from the last layer
            
        Z, a = out[l + 1] 
		
        if l == length(nn)
                
            #cumpute δᴸ
            δᴸ = a - Y'  #C_(a, Y') * map(σ_, a)
            insert!(δ, 1, δᴸ)
                
        else
            #compute δˡ
            δˡ = (nn[l + 1][1]' * δ[1]) .* map(dσ, a)
            insert!(δ, 1, δˡ)
            
        end
    end
    return δ
end

# ╔═╡ 8afdd629-5b36-4144-9663-d519b8bcd5d6
md"
Create the `gradient_descent` function that will help us to optimize the **Cost function** and find the adequate parameters for the model.
"

# ╔═╡ e001cde3-90ef-4c6b-83ee-a43c2841d1e6
function gradient_descent(nn, X, Y, α)
    
    out = forwardprop(nn, X)
    δ = backprop(nn, X, Y)
    δᵦ = copy(δ)
    
    for i in 1:length(δᵦ)
        δᵦ[i] = [mean(δᵦ[i][j,:]) for j in 1:size(δ[i])[1]]
    end

    for l in reverse(1:length(nn))
        Z, a = out[l]

        nn[l][2] -= α * δᵦ[l]
        nn[l][1] -= α * δ[l] * a'
    end
    
    return nn
end

# ╔═╡ 54268fdf-f7aa-4241-99f2-14061789821a
md"Finally, we construct the function `train` that, as the name say, it will train the model with the training set that we'll define in the next cells."

# ╔═╡ 9e3a2f1d-a939-4407-ba61-d6a12826a4d0
function train(nn, X, Y, α, n_iter)
    
    m, n = size(X)
    cost = []
    
    @showprogress for i in 1:n_iter
		
        nn = gradient_descent(nn, X, Y, α)
		
        if i == 1 || i % (n_iter/50) == 0
            append!(cost, C(forwardprop(nn, X)[end][2], Y))
        end
    end
        
    return nn, cost
end

# ╔═╡ fee50adf-36ef-4197-adca-658187874d8a
md"
#### Let's try out our Neural Network

The first step to train our model is split the dataset into **train set** and **test set**.
"

# ╔═╡ 63a96be7-e6fd-44f1-a7f3-d59053443f60
md"
We define some parameters and variables:
- `train_set_size` Sample size (percentage of the dataset) to train the model
- `indexes` Random indexes that we'll take from the data set to train the model
- `X_train` Set to train the model (is a matrix). Because we have just 13 characteristics (columns) and 303 instances (rows) in our dataset, the size of the matrix will be 
$(303\cdot train\_set\_test\times 13)$
- `y_test` Targets of the training set (is a vector)
"

# ╔═╡ ba7d9b72-94b9-43f8-b2f1-1ae7bf1a63b3
train_set_size = 0.7

# ╔═╡ 6a9bb054-e31e-4de2-9c79-96f8ebe0d4b0
indexes = sample(1:size(df)[1], Int64(round(size(df)[1]*0.7)), replace = false)

# ╔═╡ 73b3d4eb-1ebb-47d3-ad84-1833f855e9b6
X_train = Matrix(df[indexes, 1:13]);

# ╔═╡ aea84b4b-70e7-450d-8644-795d54aad534
y_train = Array(df[indexes, 14]);

# ╔═╡ c9033521-40a9-44bf-a650-3b47bf1355ed
X_test = Matrix(df[filter(x -> !(x in indexes), 1:size(df)[1]), 1:13]);

# ╔═╡ ba3aae46-aba8-4c6a-b681-59e297f9a8b7
y_test = Array(df[filter(x -> !(x in indexes), 1:size(df)[1]), 14]);

# ╔═╡ f359fa5e-ee96-48a8-bbb8-d123e3cb9101
md"
##### Starting the traing phase...
"

# ╔═╡ 296ace34-6070-4064-9b23-a8a4e4d2cbe1
p = size(X_train)[2]; #size of the inputs

# ╔═╡ 9313971a-0669-449f-848f-3c5afbbb0284
structure = [p 4 1]; # Structure of the nn

# ╔═╡ ba904a57-8c04-421a-86c0-70ac0c344eeb
nn = create_nn(structure); # Build the nn

# ╔═╡ 6c8f889a-15d2-455b-bc7b-436be98d71d4
α = 0.03 #Gradient descent parameter (nn learning ratio)

# ╔═╡ 62a34665-42d2-418a-bd95-ecde18afa844
n_iter = 1000 #Number of iterations to train the nn

# ╔═╡ 3f39ba5c-93ed-44ca-b6a9-3955f81b0f89
nn_train, cost = train(nn, X_train, y_train, α, n_iter); #training the nn

# ╔═╡ 8f552d92-1f38-4188-a56e-824e568770c6
md"
The next graph shows the evolution of the cost function through the iterations, remind we was trying to minimize it. The lower the reached value, the better fit of the trainig set. We must be careful to don't overfit the model.
"

# ╔═╡ 6318bff9-10fb-4520-86c9-5534ba839c35
begin
	lowcost = round(cost[end], digits=4)
	
	plot(0:n_iter÷50:length(cost)*(n_iter÷50) - 1, cost,
	    markershape = :circle, markersize=3,
	    color="orangered", lw=1.3,
	    title="Cost function evolution",  
	    xlabel="Iterations",
	    ylabel=L"C(W^l_{i,j})",
	    label="Reached Cost: $lowcost")
end

# ╔═╡ 26c6da6a-9182-493e-8930-11fbf2a2b333
md"
#### Let´s test our model

Once we have a model that appear to work well, we proceed to test it using the **test set** we created before.

`out` is the vector that the neural network predict to be the targets of the **train set**. We will compare it with the actual targets vector, **y_train**, and then we will able to compute the precision of the neural netowork for the **training set**.

`preds` is the vector that the neural network predict to be the targets of the **test set**. We will compare it with the actual targets vector, **y_test**, and then we will able to compute the precision of the neural netowork for the **test set**.
"

# ╔═╡ d3704bda-936c-4095-aaa5-ff205f800ba3
out = round.(forwardprop(nn, X_train)[end][2])

# ╔═╡ 26852a75-85e9-4610-a8bb-01bc3953b16b
train_precision = round(100sum([if (y_train[i] == out[i]) 1 else 
				0 end for i in 1:length(y_train)]) / length(y_train), digits=2)

# ╔═╡ a8d12553-d45e-4142-99f7-fed8788d1bd5
preds = round.(forwardprop(nn, X_test)[end][2])

# ╔═╡ 1d4e2eb3-3dd2-41f5-b109-06305910220b
test_precision = round(100sum([if (y_test[i] == preds[i]) 1 else 
				0 end for i in 1:length(y_test)]) / length(y_test), digits=2)

# ╔═╡ 4bf2010e-d750-4d0c-9373-28463495d4f4
md"
As you can see, we reached a precision of approximately $93.39\%$ for the traing set, we should think in overfitting, but our precision is pretty decent for the test set, it's about $81.32\%$
"

# ╔═╡ Cell order:
# ╟─f55a1334-0365-4099-9bbe-3b3e0ddbfd43
# ╟─05f09de2-a658-47fa-be14-2d1aed589c44
# ╟─a209e430-000b-4f08-a2a0-db08e7f71363
# ╠═60fb4aaa-fea4-41d1-b745-4e81ab616f04
# ╟─6bc223a4-665d-4d32-aa7a-0eaa56ce8145
# ╠═9f79f248-2129-4ed0-b312-fcfc13660a2d
# ╠═a03b9b05-45cf-4e08-9768-d2234bae49f9
# ╠═8fa3fc4f-8dd6-4772-bd88-51ee84b9da25
# ╟─a66ebfea-e2f4-4675-91ce-742071b18319
# ╟─d3a26b86-9b6e-449e-a6de-b9232ab185e9
# ╠═1985b05d-8ffc-41ff-8acb-ad7f37a0f7e8
# ╟─dde43202-2846-4121-afc1-8f7b7747f033
# ╠═ea3d18e0-902f-46ab-8ec6-e68467460ec3
# ╟─4452dac0-c423-46e1-bd97-2dc1039474d9
# ╠═a452c9da-54b5-4252-a104-5ec36799bb02
# ╠═cf7b9b26-81fc-4d90-88f4-939b213e5a80
# ╟─8afdd629-5b36-4144-9663-d519b8bcd5d6
# ╠═e001cde3-90ef-4c6b-83ee-a43c2841d1e6
# ╟─54268fdf-f7aa-4241-99f2-14061789821a
# ╠═9e3a2f1d-a939-4407-ba61-d6a12826a4d0
# ╟─fee50adf-36ef-4197-adca-658187874d8a
# ╟─63a96be7-e6fd-44f1-a7f3-d59053443f60
# ╟─ba7d9b72-94b9-43f8-b2f1-1ae7bf1a63b3
# ╟─6a9bb054-e31e-4de2-9c79-96f8ebe0d4b0
# ╠═73b3d4eb-1ebb-47d3-ad84-1833f855e9b6
# ╠═aea84b4b-70e7-450d-8644-795d54aad534
# ╠═c9033521-40a9-44bf-a650-3b47bf1355ed
# ╠═ba3aae46-aba8-4c6a-b681-59e297f9a8b7
# ╟─f359fa5e-ee96-48a8-bbb8-d123e3cb9101
# ╠═296ace34-6070-4064-9b23-a8a4e4d2cbe1
# ╠═9313971a-0669-449f-848f-3c5afbbb0284
# ╠═ba904a57-8c04-421a-86c0-70ac0c344eeb
# ╠═6c8f889a-15d2-455b-bc7b-436be98d71d4
# ╠═62a34665-42d2-418a-bd95-ecde18afa844
# ╠═3f39ba5c-93ed-44ca-b6a9-3955f81b0f89
# ╟─8f552d92-1f38-4188-a56e-824e568770c6
# ╟─6318bff9-10fb-4520-86c9-5534ba839c35
# ╟─26c6da6a-9182-493e-8930-11fbf2a2b333
# ╟─d3704bda-936c-4095-aaa5-ff205f800ba3
# ╟─26852a75-85e9-4610-a8bb-01bc3953b16b
# ╟─a8d12553-d45e-4142-99f7-fed8788d1bd5
# ╟─1d4e2eb3-3dd2-41f5-b109-06305910220b
# ╠═4bf2010e-d750-4d0c-9373-28463495d4f4
# ╟─bd967d50-ba6d-11eb-0200-9962ce7b1a9c
