module DirichletProcess

using Distributions, PyPlot, PDMats
using StatsFuns.logsumexp

export GW, DP
export learn_CGS


struct GW
	beta::Float64
	m::Vector{Float64}
	nu::Float64
	W::Matrix{Float64}
end

struct DP
	K::Int
	D::Int
	alpha::Float64
	cmp::Vector{GW}
end


####################
## functions
function init_S(dp::DP, X::Matrix{Float64})
	K = dp.K
	N = size(X,2)
	S = categorical_sample(ones(K)/(K), N)
	return S
end

categorical_sample(p::Vector{Float64}) = categorical_sample(p, 1)[:, 1]
function categorical_sample(p::Vector{Float64}, N::Int)
	K = length(p)
	S = zeros(K, N)
	S_tmp = rand(Categorical(p), N)
	for k in 1 : K
		S[k, find(S_tmp .== k)] = 1
	end
	return S
end

function calc_ln_pdf_ST(gw::GW, xn::Vector{Float64})
	D = size(xn,1)
	tmp = ((1.0 - D + gw.nu) / (1.0 + gw.beta)) * gw.beta * gw.W
	ln_pdf = logpdf(MvTDist(1.0 - D + gw.nu, gw.m, PDMats.PDMat(Symmetric(inv(tmp)))), xn)
	return ln_pdf
end

function sample_Sn(dp::DP, xn::Vector{Float64}, S::Matrix{Float64})
	K = dp.K
	sum_S = vec(sum(S, 2))
	tmp_ln_p = [(calc_ln_pdf_ST(dp.cmp[k], xn) + log(sum_S[k]))
			for k in 1 : K]
	push!(tmp_ln_p, (calc_ln_pdf_ST(dp.cmp[K+1], xn) + log(dp.alpha)))
	tmp_ln_p = tmp_ln_p - logsumexp(tmp_ln_p)
	sn = categorical_sample(exp.(tmp_ln_p))
	return sn
end

function update_DP(dp::DP, S::Matrix{Float64}, k::Int)
	S_tmp = S[1:size(S,1) .!= k, :]
	cmp_tmp = dp.cmp[1:size(dp.cmp,1) .!= k]
	sum_S = vec(sum(S_tmp, 2))
	return S_tmp, DP(dp.K-1, dp.D, dp.alpha, cmp_tmp)
end

function remove_stats(dp::DP, X::Matrix{Float64}, S::Matrix{Float64}, n::Int)
	K = dp.K
	# search class Sn belong to
	idx = find(S[:,n] .== 1)[1]
	# remove stats
	sum_S = vec(sum(S, 2))

	# num of data belong to the class is 0 or not
	if sum_S[idx] == 0
		S, dp = update_DP(dp, S, idx)
	else
		#############
		# あるクラスの所属数が０でない場合の処理をここに記述
		#############
		cmp = Vector{GW}()
		for k in 1 : K
			beta = dp.cmp[k].beta - S[k,n]
			m = (1.0/beta)*(dp.cmp[k].beta*dp.cmp[k].m - S[k,n]*X[:,n])
			nu = dp.cmp[k].nu - S[k,n]
			W = inv(inv(dp.cmp[k].W)
					+ dp.cmp[k].beta*dp.cmp[k].m*dp.cmp[k].m'
					- beta*m*m'
					- S[k,n]*X[:,[n]]*X[:,[n]]')
			push!(cmp, GW(beta, m, nu, W))
		end
		push!(cmp, GW(dp.cmp[K+1].beta, dp.cmp[K+1].m, dp.cmp[K+1].nu, dp.cmp[K+1].W))
		dp = DP(K, dp.D, dp.alpha, cmp)
	end

	return S, dp
end

function add_stats(dp::DP, X::Matrix{Float64}, S::Matrix{Float64}, n::Int)
	K = dp.K
	cmp = Vector{GW}()
	for k in 1 : K
		beta = dp.cmp[k].beta + S[k,n]
		m = (1.0/beta)*(dp.cmp[k].beta*dp.cmp[k].m + S[k,n]*X[:,n])
		nu = dp.cmp[k].nu + S[k,n]
		W = inv(dp.cmp[k].beta*dp.cmp[k].m*dp.cmp[k].m'
				- beta*m*m'
				+ S[k,n]*X[:,[n]]*X[:,[n]]'
				+ inv(dp.cmp[k].W))
		push!(cmp, GW(beta, m, nu, W))
	end
	push!(cmp, GW(dp.cmp[K+1].beta, dp.cmp[K+1].m, dp.cmp[K+1].nu, dp.cmp[K+1].W))
	return DP(K, dp.D, dp.alpha, cmp)
end

function add_newClass(dp::DP, sn::Vector{Float64}, S::Matrix{Float64}, n::Int)
	N = size(S, 2)
	K = dp.K

	tmp = zeros(N)
	tmp[n] = 1
	tmp = tmp[:, [1]]'
	S_tmp = vcat(S, tmp)

	cmp = Vector{GW}()
	for k in 1 : K+1
		beta = dp.cmp[k].beta
		m = dp.cmp[k].m
		nu = dp.cmp[k].nu
		W = dp.cmp[k].W
		push!(cmp, GW(beta, m, nu, W))
	end
	beta = dp.cmp[K+1].beta
	m = dp.cmp[K+1].m
	nu = dp.cmp[K+1].nu
	W = dp.cmp[K+1].W
	push!(cmp, GW(beta, m, nu, W))
	return S_tmp, DP(K+1, dp.D, dp.alpha, cmp)
end

function update_S(dp::DP, sn::Vector{Float64}, S::Matrix{Float64}, n::Int)
	k = find(sn .== 1)[1]
	if k == (dp.K + 1)
		# add new class
		S, dp = add_newClass(dp, sn, S, n)
	else
		s_tmp = zeros(dp.K)
		s_tmp[k] = 1.0
		S[:, n] = s_tmp
	end
	return S, dp
end

function sample_S_CGS(dp::DP, X::Matrix{Float64}, S::Matrix{Float64})
	N = size(X, 2)

	for n in 1 : N
		S, dp = remove_stats(dp, X, S, n)
		sn = sample_Sn(dp, X[:,n], S)
		S, dp = update_S(dp, sn, S, n)
		dp = add_stats(dp, X, S, n)
	end
	return S, dp
end

function first_update(dp::DP, X::Matrix{Float64}, S::Matrix{Float64})
	K = dp.K
	D = dp.D
	sum_S = sum(S, 2)
	cmp = Vector{GW}()

	for k in 1 : K
		beta = sum_S[k] + dp.cmp[k].beta
		m = (1.0/beta)*(vec(sum(X[:,find(S[k, :].==1)], 2))+ dp.cmp[k].beta*dp.cmp[k].m)
		nu = sum_S[k] + dp.cmp[k].nu
		W = inv(X*diagm(S[k,:])*X'
					+ dp.cmp[k].beta*dp.cmp[k].m*dp.cmp[k].m'
					+ inv(dp.cmp[k].W)
					- beta*m*m')
		push!(cmp, GW(beta, m, nu, W))
	end
	push!(cmp, GW(dp.cmp[K+1].beta, dp.cmp[K+1].m, dp.cmp[K+1].nu, dp.cmp[K+1].W))
	return DP(K, D, dp.alpha, cmp)
end


####################
## main algorithm
function learn_CGS(X::Matrix{Float64}, prior_dp::DP, max_iter::Int)
	S = init_S(prior_dp, X)
	dp = first_update(prior_dp, X, S)

	for i in 1 : max_iter
		S, dp = sample_S_CGS(dp, X, S)
	end

	return S, dp
end

end