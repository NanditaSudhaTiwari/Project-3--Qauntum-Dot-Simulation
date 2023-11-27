import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, m_e, k
from scipy.special import logsumexp

# Harmonic oscillator potential for a quantum dot
def harmonic_oscillator_potential(x, omega):
    return 0.5 * m_e * omega**2 * x**2

# Kinetic energy operator
def kinetic_energy_operator(psi, dx):
    h = hbar / dx
    return -h**2 / (2 * m_e) * np.gradient(np.gradient(psi, dx), dx)

# Hamiltonian operator for the quantum dot simulation
def hamiltonian_operator(psi, x, omega, dx):
    potential = harmonic_oscillator_potential(x, omega)
    kinetic = kinetic_energy_operator(psi, dx)
    return kinetic + np.diag(potential) @ psi

# Eigenstates of the quantum dot
def quantum_dot_eigenstates(x, omega, n_states):
    eigenstates = np.zeros((len(x), n_states))
    for n in range(1, n_states + 1):
        eigenstates[:, n-1] = np.sqrt(2 / (2**n * np.math.factorial(n))) * (m_e * omega / (np.pi * hbar))**0.25 * np.exp(-m_e * omega * x**2 / (2 * hbar) * (-1)**n) * np.polynomial.hermite.hermval(np.sqrt(m_e * omega / hbar) * x, [0] * n + [1])
    return eigenstates

# Coefficients for the linear combination of eigenstates
def linear_combination_coefficients(n_states):
    return np.random.rand(n_states)  # Modify as needed for specific coefficients

# Linear combination of eigenstates
def linear_combination(x, omega, n_states):
    eigenstates = quantum_dot_eigenstates(x, omega, n_states)
    coefficients = linear_combination_coefficients(n_states)
    psi = eigenstates @ coefficients
    return psi / np.linalg.norm(psi)

# Function to normalize the wavefunction
def normalize_psi(psi, dx):
    norm = np.sqrt(np.trapz(np.abs(psi)**2, dx=dx))
    return psi / norm if norm != 0 else psi

# Metropolis-Hastings algorithm for quantum dot simulation
def metropolis_hastings_quantum_dot(psi, x, omega, iterations, proposal_scale, temperature):
    dx = x[1] - x[0]
    beta = 1 / (k * temperature)
    psi_history = [psi.copy()]

    for _ in range(iterations):
        proposed_psi = propose_move(psi, proposal_scale, dx)
    
        # Normalize the wavefunctions
        psi = normalize_psi(psi, dx)
        proposed_psi = normalize_psi(proposed_psi, dx)

        current_energy = calculate_energy(psi, x, omega, dx)
        proposed_energy = calculate_energy(proposed_psi, x, omega, dx)

        max_energy = max(current_energy, proposed_energy)
        
        # Calculate acceptance ratio
        if np.exp(-beta * max_energy) == 0:
            acceptance_ratio = 0.0
        else:
            acceptance_ratio = np.exp(-beta * (proposed_energy - current_energy - max_energy + np.log(np.exp(-beta * max_energy))))

        # Accept or reject the move
        if np.random.rand() < acceptance_ratio:
            psi = proposed_psi
        
        psi_history.append(psi.copy())

    return np.array(psi_history)

# Proposal function for updating wavefunction
def propose_move(psi, proposal_scale, dx):
    return normalize_psi(psi + proposal_scale * np.random.randn(len(psi)) * np.sqrt(dx), dx)

# Calculate energy for the quantum dot
def calculate_energy(psi, x, omega, dx):
    kinetic_energy = np.sum(np.conj(psi) * kinetic_energy_operator(psi, dx)) * dx
    potential_energy = np.sum(np.conj(psi) * harmonic_oscillator_potential(x, omega) * psi) * dx
    return kinetic_energy + potential_energy

# Function to visualize the wavefunction evolution over iterations
def visualize_wavefunction_evolution(psi_history, x, omega, n_states):
    plt.figure(figsize=(15, 8))

    for psi in psi_history[::100]:
        plt.plot(x, np.abs(psi)**2, label=f'Iteration {len(psi)}')

    # Plot the individual eigenstates
    eigenstates = quantum_dot_eigenstates(x, omega, n_states)
    for i in range(n_states):
        plt.plot(x, np.abs(eigenstates[:, i])**2, linestyle='--', label=f'Eigenstate {i + 1}')

    plt.xlabel('Position (m)')
    plt.ylabel('Probability Density')
    plt.title('Wavefunction Evolution Over Iterations')
    plt.legend()
    plt.legend(loc='upper right')
    plt.show()

# Function to visualize the final wavefunctions for different parameters
def visualize_final_wavefunctions(parameter_sets, x, omega, iterations, n_states):
    plt.figure(figsize=(15, 8))

    for params in parameter_sets:
        psi_history = metropolis_hastings_quantum_dot(params['psi_initial'], x, omega, iterations, params['proposal_scale'], params['temperature'])
        plt.plot(x, np.abs(psi_history[-1])**2, label=f'Scale={params["proposal_scale"]}, Temp={params["temperature"]} K')

    # Plot the individual eigenstates
    eigenstates = quantum_dot_eigenstates(x, omega, n_states)
    for i in range(n_states):
        plt.plot(x, np.abs(eigenstates[:, i])**2, linestyle='--', label=f'Eigenstate {i + 1}')

    plt.xlabel('Position (m)')
    plt.ylabel('Probability Density')
    plt.title('Final Wavefunctions for Different Parameters')
    plt.legend()
    plt.legend(loc='upper right')
    plt.show()

# Function to visualize the energy convergence over iterations
def visualize_energy_convergence(psi_history, x, omega, iterations):
    energies = [calculate_energy(psi, x, omega, x[1] - x[0]) for psi in psi_history]

    plt.figure(figsize=(15, 5))
    plt.plot(range(iterations + 1), energies, label='Energy Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('Energy Convergence Over Iterations')
    plt.legend()
    plt.legend(loc='upper right')
    plt.show()

# Function to visualize the acceptance ratio over iterations
def visualize_acceptance_ratio(psi_history, x, omega, iterations, temperature):
    beta = 1 / (k * temperature)
    acceptance_ratios = []

    for i in range(iterations):
        current_energy = calculate_energy(psi_history[i], x, omega, x[1] - x[0])
        proposed_energy = calculate_energy(psi_history[i + 1], x, omega, x[1] - x[0])

        acceptance_ratio = np.exp(-beta * (proposed_energy - current_energy))
        acceptance_ratios.append(acceptance_ratio)

    plt.figure(figsize=(15, 5))
    plt.plot(range(1, iterations + 1), acceptance_ratios, label='Acceptance Ratio')
    plt.xlabel('Iteration')
    plt.ylabel('Acceptance Ratio')
    plt.title('Acceptance Ratio Over Iterations')
    plt.legend()
    plt.legend(loc='upper right')
    plt.show()

# Gelman-Rubin statistic for convergence testing
def gelman_rubin_statistic(chains):
    num_chains, chain_length, num_params = chains.shape
    means = np.mean(chains, axis=1)
    overall_mean = np.mean(means, axis=0)
    B = chain_length / (num_chains - 1) * np.sum((means - overall_mean)**2, axis=0)
    W = np.mean(np.var(chains, axis=1), axis=0)
    var_hat = (chain_length - 1) / chain_length * W + B / chain_length
    R_hat = np.sqrt(var_hat / W)
    return R_hat

# Autocorrelation diagnostic for convergence testing
def autocorrelation_diagnostic(chain):
    n = len(chain)
    autocorr = np.correlate(chain, chain, mode='full') / np.var(chain)
    autocorr = autocorr[n-1:]  # Take only positive lags
    autocorr /= np.arange(n, 0, -1)  # Normalize by lag
    return autocorr / autocorr[0]

# Function to check convergence using Gelman-Rubin and autocorrelation diagnostics
def check_convergence(chains):
    R_hat = gelman_rubin_statistic(chains)
    print(f'Gelman-Rubin Statistic (R_hat): {R_hat}')

    for i, chain in enumerate(chains[0].T):
        autocorr = autocorrelation_diagnostic(chain)
        plt.plot(autocorr, label=f'Parameter {i + 1}')

    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Diagnostic')
    plt.legend()
    plt.legend(loc='upper right')
    plt.show()

# Metropolis-Hastings algorithm for quantum dot simulation with convergence testing
def metropolis_hastings_quantum_dot_convergence(psi, x, omega, iterations, proposal_scale, temperature):
    dx = x[1] - x[0]
    beta = 1 / (k * temperature)
    psi_history = [psi.copy()]

    for _ in range(iterations):
        proposed_psi = propose_move(psi, proposal_scale, dx)

        # Normalize the wavefunctions
        psi = normalize_psi(psi, dx)
        proposed_psi = normalize_psi(proposed_psi, dx)

        current_energy = calculate_energy(psi, x, omega, dx)
        proposed_energy = calculate_energy(proposed_psi, x, omega, dx)

        max_energy = max(current_energy, proposed_energy)

        # Calculate acceptance ratio
        if np.exp(-beta * max_energy) == 0:
            acceptance_ratio = 0.0
        else:
            acceptance_ratio = np.exp(-beta * (proposed_energy - current_energy - max_energy + np.log(np.exp(-beta * max_energy))))

        # Accept or reject the move
        if np.random.rand() < acceptance_ratio:
            psi = proposed_psi

        psi_history.append(psi.copy())

    return np.array(psi_history)

# Function to validate convergence using Gelman-Rubin and autocorrelation diagnostics
def validate_convergence(psi_history, x, omega, iterations, proposal_scale, temperature, n_states):
    visualize_wavefunction_evolution(psi_history, x, omega, n_states)
    visualize_energy_convergence(psi_history, x, omega, iterations)
    visualize_acceptance_ratio(psi_history, x, omega, iterations, temperature)

# Main function with convergence testing
def main_with_convergence():
    # Quantum dot parameters
    omega = 1e12    # Angular frequency for the harmonic oscillator potential (adjust as needed)
    n_states = 3    # Number of eigenstates to consider
    x = np.linspace(-1e-9, 1e-9, 1000)  # Spatial grid
    psi_initial = linear_combination(x, omega, n_states)  # Initial wavefunction as a linear combination of eigenstates

    # MCMC parameters
    iterations = 5000
    proposal_scale_values = [0.01, 0.02, 0.03]
    temperature_values = [300, 500, 1000]

    # Visualize wavefunction evolution over iterations
    psi_history = metropolis_hastings_quantum_dot(psi_initial, x, omega, iterations, proposal_scale_values[0], temperature_values[0])
    visualize_wavefunction_evolution(psi_history, x, omega, n_states)

    # Visualize final wavefunctions for different parameters
    parameter_sets = [{'psi_initial': psi_initial, 'proposal_scale': 0.02, 'temperature': 300},
                      {'psi_initial': psi_initial, 'proposal_scale': 0.02, 'temperature': 500},
                      {'psi_initial': psi_initial, 'proposal_scale': 0.02, 'temperature': 1000}]
    visualize_final_wavefunctions(parameter_sets, x, omega, iterations, n_states)

    # Visualize energy convergence over iterations
    psi_history = metropolis_hastings_quantum_dot(psi_initial, x, omega, iterations, proposal_scale_values[0], temperature_values[0])
    visualize_energy_convergence(psi_history, x, omega, iterations)

    # Visualize acceptance ratio over iterations
    psi_history = metropolis_hastings_quantum_dot(psi_initial, x, omega, iterations, proposal_scale_values[0], temperature_values[0])
    visualize_acceptance_ratio(psi_history, x, omega, iterations, temperature_values[0])

    # Run MCMC with convergence testing
    chains = []
    for proposal_scale, temperature in zip(proposal_scale_values, temperature_values):
        psi_history = metropolis_hastings_quantum_dot_convergence(psi_initial, x, omega, iterations, proposal_scale, temperature)
        chains.append(psi_history)

    # Validate convergence using Gelman-Rubin and autocorrelation diagnostics
    for psi_history, proposal_scale, temperature in zip(chains, proposal_scale_values, temperature_values):
        print(f'\nConvergence Testing for Proposal Scale: {proposal_scale}, Temperature: {temperature} K')
        validate_convergence(psi_history, x, omega, iterations, proposal_scale, temperature, n_states)

    # Check convergence using Gelman-Rubin and autocorrelation diagnostics
    check_convergence(np.array(chains))

if __name__ == "__main__":
    main_with_convergence()
