import matplotlib.pyplot as plt
import numpy as np

k = 1
m = 1
x0 = 0
v0 = 1

t = 0
tmax = 20
dt = 0.01

def F(x):
    return -k/m*x

def semi_implicit(x, v):
    v_new = v +F(x)*dt
    x_new = x + v_new*dt
    return x_new, v_new
    
def explicit(x,v):
    v_new = v +F(x)*dt
    x_new = x + v*dt
    return x_new, v_new
    
def runge_kutta(x,v):
    v_new = v + F(x)*dt
    x_new = x + dt/2*(v+v_new)
    return x_new, v_new
    
def energy(x,v):
    return 0.5*k*x**2+0.5*m*v**2
    
    
def run(x0, v0, integrator):
    t=0
    t_list = [0,]
    x_list = [x0,]
    v_list = [v0,]
    E_list = [energy(x0,v0),]
    while t<tmax:
        x_new, v_new = integrator(x_list[-1], v_list[-1])
        t += dt
        
        x_list.append(x_new)
        v_list.append(v_new)
        E_list.append(energy(x_new,v_new))
        t_list.append(t)
    return t_list, x_list, v_list, np.array(E_list)

t_list, x_explicit, v_explicit, E_explicit = run(x0, v0, explicit)
t_list, x_semi_implicit, v_semi_implicit, E_semi_implicit = run(x0,v0,semi_implicit)
t_list, x_RK, v_RK, E_RK = run(x0, v0, runge_kutta)

E_initial_explicit = E_explicit[0]
E_initial_semi_implicit = E_semi_implicit[0]
E_initial_RK = E_RK[0]


plt.plot(t_list, E_explicit, label="Explicit Euler")
plt.plot(t_list, E_semi_implicit, label="Symplectic Euler")
plt.plot(t_list, E_RK, label="2nd-order RK")

plt.hlines(E_RK[0], t_list[0], t_list[-1], label="Conserved total energy", color='gray', ls='--')
plt.legend()
plt.title("Total Energy")
plt.xlabel("Time")
plt.ylabel("Total Energy")

plt.show()

#Some stuff to plot fractional differences from the initial total energy
plt.plot(t_list, np.abs(E_explicit-E_initial_explicit)/E_initial_explicit, label="Explicit Euler")
plt.plot(t_list, np.abs(E_semi_implicit-E_initial_semi_implicit)/E_initial_semi_implicit, label="Symlectic Euler")
plt.plot(t_list, np.abs(E_RK-E_initial_RK)/E_initial_RK, label="2nd-order RK")
plt.legend()
plt.title("Fractional Difference in Energy")
plt.xlabel("Time")
plt.ylabel(r"$\frac{E(t)-E(0)}{E(0)}$")
plt.yscale('log')
plt.show()
