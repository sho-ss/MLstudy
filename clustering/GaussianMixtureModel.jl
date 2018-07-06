module GaussianMixtureModel

using Distributions
using PDMats
using StatsFuns.logsumexp

export GW, Gauss, GMM, BGMM
export learn_GS, learn_CGS

################
## types
struct GW
	beta::Float64
	m::Vector{Float64}
	nu::Float64
	W::Matrix{Float64}
end

struct Gauss
	mu::Vector{Float64}
	Lambda::Matrix{Float64}
end

struct GMM
	K::Int
	D::Int
	phi::Vector{Float64}
	cmp::Vector{Gauss}
end

struct BGMM
	K::Int
	D::Int
	alpha::Vector{Float64}
	cmp::Vector{GW}
end


#################
## functions
categorical_sample(p::Vector{Float64}) = categorical_sample(p, 1)[:, 1]
function categorical_sample(p::Vector{Float64}, N::Int)
	K = length(p)
	S = zeros(K, N)
	S_tmp = rand(Categorical(p), N)
	for k in 1 : K
		S[k, find(S_tmp.==k)] = 1
	end
	return S
end

function init_S(bgmm::BGMM, X::Matrix{Float64})
	K = bgmm.K
	N = size(X,2)
	S = categorical_sample(ones(K)/K, N)
	return S
end

function update_BGMM(bgmm::BGMM, X::Matrix{Float64}, S::Matrix{Float64})
	K = bgmm.K
	D = bgmm.D
	sum_S = sum(S, 2)
	cmp = Vector{GW}()

	for k in 1 : K
		beta = sum_S[k] + bgmm.cmp[k].beta
		m = (1.0/beta)*(vec(sum(X[:,find(S[k, :].==1)], 2))+ bgmm.cmp[k].beta*bgmm.cmp[k].m)
		nu = sum_S[k] + bgmm.cmp[k].nu
		W = inv(X*diagm(S[k,:])*X'
					+ bgmm.cmp[k].beta*bgmm.cmp[k].m*bgmm.cmp[k].m'
					+ inv(bgmm.cmp[k].W)
					- beta*m*m')
		push!(cmp, GW(beta, m, nu, W))
	end
	alpha = [sum_S[k]+bgmm.alpha[k] for k in 1:K]
	return BGMM(K, D, alpha, cmp)
end

function sample_GMM(bgmm::BGMM)
	K = bgmm.K
	D = bgmm.D
	cmp = Vector{Gauss}()
	for c in bgmm.cmp
		Lambda = rand(Wishart(c.nu, PDMats.PDMat(Symmetric(c.W))))
		mu = rand(MvNormal(c.m, PDMats.PDMat(Symmetric(inv(c.beta*Lambda)))))
		push!(cmp, Gauss(mu, Lambda))
	end
	phi = rand(Dirichlet(bgmm.alpha))
	return GMM(K, D, phi, cmp)
end

function sample_S(gmm::GMM, X::Matrix{Float64})
	K = gmm.K
	N = size(X, 2)
	S = zeros(K, N)

	tmp = [0.5*logdet(gmm.cmp[k].Lambda) + log(gmm.phi[k]) for k in 1:K]
	for n in 1 : N
		tmp_ln_phi = [(tmp[k]
						- 0.5*trace(gmm.cmp[k].Lambda*(X[:,n] - gmm.cmp[k].mu)*(X[:,n] - gmm.cmp[k].mu)'))
						for k in 1:K]
		tmp_ln_phi = tmp_ln_phi - logsumexp(tmp_ln_phi)
		S[:, n] = categorical_sample(exp.(tmp_ln_phi))
	end
	return S
end


#############
## used for CGS
function calc_ln_pdf_ST(gw::GW, xn::Vector{Float64})
	D = size(xn,1)
	tmp = ((1.0 - D + gw.nu) / (1.0 + gw.beta)) * gw.beta * gw.W
	ln_pdf = logpdf(MvTDist(1.0 - D + gw.nu, gw.m, PDMats.PDMat(Symmetric(inv(tmp)))), xn)
	return ln_pdf
end


remove_stats(bgmm::BGMM, Xn::Matrix{Float64}, Sn::Matrix{Float64}) = add_stats(bgmm, Xn, -Sn)

function add_stats(bgmm::BGMM, Xn::Matrix{Float64}, Sn::Matrix{Float64})
	alpha = bgmm.alpha
	K = bgmm.K
	D = bgmm.D
	cmp = Vector{GW}()

	for k in 1 : K
		beta = bgmm.cmp[k].beta + Sn[k]
		m = (1.0/beta) * (bgmm.cmp[k].beta*bgmm.cmp[k].m + Sn[k]*vec(Xn))
		nu = bgmm.cmp[k].nu + Sn[k]
		W = inv(Sn[k]*Xn*Xn'
					+ bgmm.cmp[k].beta*bgmm.cmp[k].m*bgmm.cmp[k].m'
					- beta*m*m'
					+ inv(bgmm.cmp[k].W))
		push!(cmp, GW(beta, m, nu, W))
	end
	alpha_tmp = alpha .+ vec(Sn)
	return BGMM(K, D, alpha_tmp, cmp)
end

function sample_sn(bgmm::BGMM, xn::Vector{Float64})
	tmp_ln_p = [(calc_ln_pdf_ST(bgmm.cmp[k], xn) + log(bgmm.alpha[k]))
			for k in 1 : bgmm.K]
	tmp_ln_p = tmp_ln_p - logsumexp(tmp_ln_p)
	sn = categorical_sample(exp.(tmp_ln_p))
	return sn
end

function sample_S_CGS(bgmm::BGMM, X::Matrix{Float64}, S::Matrix{Float64})
	K = bgmm.K
	D = bgmm.D
	N = size(X, 2)
	for n in 1 : N
		# remove stats w.r.t X[:,n]
		bgmm = remove_stats(bgmm, X[:,[n]], S[:,[n]])
		S[:, n] = sample_sn(bgmm, X[:,n])
		# add stats w.r.t X[:, n]
		bgmm = add_stats(bgmm, X[:,[n]], S[:,[n]])
	end
	return S, bgmm
end



#################
## main algorithm
function learn_GS(X::Matrix{Float64}, prior_bgmm::BGMM, max_iter::Int)
	# initialization
	S = init_S(prior_bgmm, X)
	bgmm = update_BGMM(prior_bgmm, X, S)

	for i in 1 : max_iter
		gmm = sample_GMM(bgmm)
		S = sample_S(gmm, X)
		bgmm = update_BGMM(prior_bgmm, X, S)
	end

	return S, bgmm
end


function learn_CGS(X::Matrix{Float64}, prior_bgmm::BGMM, max_iter::Int)
	# initialization
	S = init_S(prior_bgmm, X)
	bgmm = update_BGMM(prior_bgmm, X, S)

	for i in 1 : max_iter
		S, bgmm = sample_S_CGS(bgmm, X, S)
		#bgmm = update_BGMM(prior_bgmm, X, S)
	end
	return S, bgmm
end


##############
## for test
function sample_data(gmm::GMM, N::Int)
    X = zeros(gmm.D, N)
    S = categorical_sample(gmm.phi, N)
    for n in 1 : N
        k = indmax(S[:, n])
        X[:,n] = rand(MvNormal(gmm.cmp[k].mu, PDMats.PDMat(Symmetric(inv(gmm.cmp[k].Lambda)))))
    end
    return X, S
end

end