excludedim(a, dim)= (a[1:dim-1],a[dim+1:end])

# takes SP activations in 1 timestep
sparseness(aᵢ)= count(aᵢ)/length(aᵢ)

"""
`entropySP(aₜᵢ; tdim=1)` given a history of SP activations in time returns the SP's total entropy.
 - `t` is assumed to be the 1st dimension. If `t` is another dimension, set `tdim`
"""
entropySP(aₜᵢ; tdim=1)= begin
  # should take a vector in time foreach minicolumn
  entropy_lim(aₜ)= aₜ ≈ 0 ? 0.0 :
    aₜ ≈ 1 ? 1.0 : entropyf(aₜ)
  # aₜᵢ:= (t,i,j,...)
  # @view aₜᵢ[:,i,j,...]
  c= Iterators.product((excludedim(size(aₜᵢ), tdim).|> CartesianIndices)...)
  mapreduce(+, c) do (pre,post)
    entropy_lim(sparseness(@view aₜᵢ[pre,:,post]))
  end / length(c)
end
entropyf(x)= -x*log2(x) -(1-x)*log2(1-x)

"""
`addnoise(sdr,k)` flips `k ∈ [0,1]`- fraction of random active bits of the SDR and the same amount of inactive,
adding k-level noise while maintaining sparsity.
"""
addnoise(sdr,k)= begin
  k<0 ? k=0 : ( k>1 ? k=1 : nothing)
  nFlip= min( round(Int,k*count(sdr)), length(sdr) - count(sdr) )
  act= (sdr|> findall|> shuffle)[1:nFlip]
  inact= (.~sdr|> findall|> shuffle)[1:nFlip]
  flip!(x,i)= begin
    @views x[i].= .~x[i]
    x
  end
  @> sdr copy flip!(act) flip!(inact)
end

noiserobustness(sp::SpatialPooler; trials=12)= begin
  @unpack szᵢₙ = sp.params
  k_step= 20
  trial(i)= begin
    z= bitrand(szᵢₙ)
    zⁿ(k)= addnoise(z,k)
    a= sp(z)
    aⁿ(k)= sp(zⁿ(k))
    # ∫{0..1} f(k)dk
    mapreduce(+, 0:1/k_step:1) do k
      count(a .& aⁿ(k)) / count(a)
    end / k_step
  end
  mapreduce(trial, +, 1:trials)/trials
end

"""
`stabilitySP(aᵢ,paᵢ)` given a vector `aᵢ` of SP activations for a set of test inputs at
time j, and the activations on the same set of test inputs at a previous time k `paᵢ`,
shows a metric of how much the SP's response has changed from training.
`aᵢ` dims: {activation, samples}
"""
stabilitySP(aᵢ,paᵢ)=
  mapreduce(+, 1:size(aᵢ,2)) do i
    @views count(aᵢ[:,i] .& paᵢ[:,i]) / count(paᵢ[:,i])
  end / size(aᵢ,2)