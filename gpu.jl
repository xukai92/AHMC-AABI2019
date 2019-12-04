logdensity_f(x) = logpdf(flow_model, reshape(x, 784, 1))[1]
# Define gradient function via reverse AD
function grad_f(x)
    val, back = Tracker.forward(logdensity_f, x)
    grad = back(1)
    return (Tracker.data(val), Float32.(Tracker.data(grad[1][:,1])))
end