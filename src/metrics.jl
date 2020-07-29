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
    aₜ ≈ 1 ? 0.0 : entropyf(aₜ)
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

"""
`(metric, changeCurve) = noiserobustness(sp::SpatialPooler, data; trials=15)` runs repeated experiments on the SP
with increasingly distorted stimuli to gauge the noise robustness.
"""
noiserobustness(sp::SpatialPooler, data; trials=10)= begin
  @unpack szᵢₙ = sp.params
  zⁿ(z,k)= addnoise(z,k)
  aⁿ(z,k)= sp(zⁿ(z,k))
  k_step= 40; sampleSize= floor(Int,size(data,2)/trials)
  changeWithNoise= zeros(k_step+1,trials)
  sampleIdx= randperm(size(data,2))
  trial(i)= begin
    zset= data[:,sampleIdx[(i-1)*sampleSize+1 : i*sampleSize]]
    # ∫{0..1} f(k)dk
    mapreduce(+, enumerate(0:1/k_step:1)) do (kᵢ,k)
      changeWithNoise[kᵢ,i]= mapreduce(+, 1:sampleSize) do s
        z= zset[:,s]; a= sp(z)
        count(a .& aⁿ(z,k)) / count(a)
      end / sampleSize
      changeWithNoise[kᵢ,i]
    end / (k_step+1)
  end
  result= map(trial, 1:trials)
  mapslices(mean,changeWithNoise,dims=[2])
  (result, changeWithNoise[:,1])
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