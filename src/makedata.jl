"""
`randomSDR()` creates a dataset of N SDRs with sparsity between 2-20%
"""
function randomSDR(n)
  len= 32*32
  data= falses(len,n)
  for i= 1:n
    k= floor(Int, (rand()*0.18 + 0.02) * len)  # number of 1s
    @views data[ randperm(len)[1:k] ,i].= true
  end
  data
end