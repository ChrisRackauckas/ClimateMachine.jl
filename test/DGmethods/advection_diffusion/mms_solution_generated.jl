const ν_exact = 1//100
const a_exact = 1000000
const b_exact = 10000
ρ_g(t, φ, θ, r) = sin(θ)^4*sin(φ)^2*sin(pi*(r/10000 - 100))*cos(t) + 3
Sρ_g(t, φ, θ, r) = (-50*r^2*sin(t)*sin(θ)^2*sin(φ)^2 + 10*sin(θ)^2*sin(φ)^2*cos(t) - 6*sin(φ)^2*cos(t) - cos(t))*sin(θ)^2*sin(pi*r/10000)/(50*r^2)