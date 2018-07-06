using PyPlot, Distributions, PyCall
push!(LOAD_PATH, ".")
import GaussianMixtureModel

##############
## plot
function plot_2D(X, S)
	K = size(S, 1)
	for k in 1 : K
		idxs = find(S[k, :] .== 1)
		println(length(idxs))
		println(S[k, :])
		color = [k for _ in 1 : length(idxs)]
		scatter(X[1, idxs], X[2, idxs], c=color)
	end
	savefig("test.png")
end

function visualize_2D(X::Matrix{Float64}, S::Matrix{Float64}, S_est::Matrix{Float64}, text)
    cmp = get_cmap("jet")

    K1 = size(S, 1)
    K2 = size(S_est, 1)
    col1 = [pycall(cmp.o, PyAny, Int(round(val)))[1:3] for val in linspace(0,255,K1)]    
    col2 = [pycall(cmp.o, PyAny, Int(round(val)))[1:3] for val in linspace(0,255,K2)]    

    f, (ax1, ax2) = subplots(1,2,num=text)
    f[:clf]()
    f, (ax1, ax2) = subplots(1,2,num=text)

    for k in 1 : K1
        ax1[:scatter](X[1, S[k,:].==1], X[2, S[k,:].==1], color=col1[k])
    end
    ax1[:set_title]("truth")

    for k in 1 : K2
        ax2[:scatter](X[1, S_est[k,:].==1], X[2, S_est[k,:].==1], color=col2[k])
    end

    ax2[:set_title]("estimation")

    savefig("test.png")
end

#############
## main

function create_data(K)
	# 多次元ガウス分布からのサンプリング
	S = zeros(K, 200)

	mu1 = Float64[2, 5]
	mu2 = Float64[-3, 2]
	sig = (fill(1.0, (2, 2)) + Diagonal([1, 1]))*0.3
	x1 = rand(MvNormal(mu1, sig), 50)
	x2 = rand(MvNormal(mu2, sig), 50)
	mu3 = Float64[1, -5]
	sig2 = Float64[1.5 1.0; 1.0 1.5]
	x3 = rand(MvNormal(mu3, sig2), 100)

	X = hcat(hcat(x1, x2), x3)
	for n in 1 : 200
		if n<=50
			S[1, n] = 1
		elseif 51<=n<=100
			S[2, n] = 1
		else
			S[3, n] = 1
		end
	end
	return X, S
end

function test()
	# set hyperparam
	K = 3
	X, S = create_data(K)
	D = size(X, 1)

	alpha = 100.0 * ones(K)
	beta = 1.0
	m = zeros(D)
	nu = D + 1.0
	W = eye(D)

	tic()
	cmp = [GaussianMixtureModel.GW(beta, m, nu, W) for _ in 1:K]
	bgmm = GaussianMixtureModel.BGMM(K, D, alpha, cmp)

	N = 200
	gmm = GaussianMixtureModel.sample_GMM(bgmm)
	#println(gmm.phi)
	X, S = GaussianMixtureModel.sample_data(gmm, N)

	print(size(X))

	# inference
	max_iter = 200
	# Gibbs Sampling
	#S_est, bgmm = GaussianMixtureModel.learn_GS(X, bgmm, max_iter)
	# Collapsed Gibbs Sampling
	S_est, bgmm = GaussianMixtureModel.learn_CGS(X, bgmm, max_iter)
	toc()
	#plot_2D(X, S)
	visualize_2D(X, S, S_est, "2D plot")
end

test()


