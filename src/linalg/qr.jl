function myqr(A)
    m = size(A,1)
    n = size(A,2)
    R = copy(A)
    Q = Matrix{Float64}(I, m, m)
    for k in 1:n
        u  = R[k:end,k]
        u0 = u[1]
        u0 = u0 + sign(u0)*norm(u)
        u  = u/norm(u)
        R[k:end,k:end] = R[k:end,k:end] - 2*u*(u'*R[k:end,k:end])
        Q[k:end,:]     = Q[k:end,:]     - 2*u*(u'*Q[k:end,:])
    end
    Q, R
end
