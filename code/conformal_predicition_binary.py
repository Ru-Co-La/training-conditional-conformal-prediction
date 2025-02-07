import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm
from sympy.printing.pretty.pretty_symbology import line_width

# Use Computer Modern fonts for math and text
plt.rcParams['font.serif'] = ['Computer Modern'] # Set serif font to Computer Modern
plt.rcParams['mathtext.fontset'] = 'cm'         # Use Computer Modern for math

# Parameters
E = np.linspace(0, 1, 100 + 1)
E = E[1:-1]
epsilon = (1 + 2/3)/2
b_greater_than_E = (E + (E+0.01))/2 #5/100*(1-E) + E
b_lower_than_E = (E + (E-0.01))/2 #95/100*E
M = int(5e4) # Number of calibration sets
L = int(5e4) # Number of test sets
N = 2

# Compute PAC guarantees using conformal prediction for every E

P_S_E_array = np.zeros(len(E))
rate_of_double_label_predictions_array = np.zeros(len(E))
rate_of_single_label_predictions_array = np.zeros(len(E))

# Loop with b > E. Then P_S_E is b^2
for i, b in enumerate(b_greater_than_E):

    # Create M pairs of Bernoulli random variables with parameter b
    cal_set_collection = np.random.binomial(1, b, (M, N))
    successful_confidence_counter = 0
    number_of_double_label_predictions = 0

    desc = str("Computing PAC guarantees for E = " + str(E[i]) + " and b = " + str(b))
    # Construct the conformal prediction set INP
    for cal_set in tqdm.tqdm(cal_set_collection, desc=desc):

        INP = set()

        # Happens with prob b^2
        # Add 0 and 1 to INP
        if all(cal_set):
            INP = {0, 1}
        # Happens with prob 1 - b^2
        # Add 0 to INP
        else:
            INP = {0}
        INP_size = len(INP)

        # INP has all the values
        if INP_size == 2:
            successful_confidence_counter += 1
            number_of_double_label_predictions += 1
            continue

        # INP has only 0
        test_set_collection = np.random.binomial(L , b)
        # count number of 0s
        number_of_zeros = L-test_set_collection
        successful_prediction_counter = number_of_zeros

        coverage = successful_prediction_counter/L

        if coverage >= 1 - E[i]:
            successful_confidence_counter += 1

    P_S_E = successful_confidence_counter/M
    rate_of_double_label_predictions = number_of_double_label_predictions/M
    rate_of_single_label_predictions = P_S_E - rate_of_double_label_predictions

    P_S_E_array[i] = P_S_E
    rate_of_double_label_predictions_array[i] = rate_of_double_label_predictions
    rate_of_single_label_predictions_array[i] = rate_of_single_label_predictions

# Plot the PAC guarantees
plt.figure(figsize=(8, 6))
# Fill areas under the curve with specified colors
plt.fill_between(E, 0, rate_of_single_label_predictions_array, color='red', alpha=0.3, label='$\mathbb{P}^2(S_E \cap \Gamma^{\epsilon}=\overline{Q})$')
plt.fill_between(E, rate_of_single_label_predictions_array, P_S_E_array, color='blue', alpha=0.3, label='$\mathbb{P}^2(S_E \cap \Gamma^{\epsilon}=\mathbf{Z})$')
# Plot E vs E^2
plt.plot(E, E**2, label="$E^2$", color='black', linewidth=2)
# Plot P_S_E_array
plt.plot(E, P_S_E_array, label="$\mathbb{P}^2(S_E)$", color='red', linewidth=2)
plt.xlabel("$E$")
plt.ylabel("Probability")
plt.title("PAC guarantees using conformal prediction")
plt.legend()
plt.grid()
plt.show()



P_S_E_array = np.zeros(len(E))
rate_of_double_label_predictions_array = np.zeros(len(E))
rate_of_single_label_predictions_array = np.zeros(len(E))

# Loop with b <= E. Then P_S_E is 1
for i, b in enumerate(b_lower_than_E):

    # Create M pairs of Bernoulli random variables with parameter b
    cal_set_collection = np.random.binomial(1, b, (M, N))
    successful_confidence_counter = 0
    number_of_double_label_predictions = 0

    desc = str("Computing PAC guarantees for E = " + str(E[i]) + " and b = " + str(b))

    # Construct the conformal prediction set INP
    for cal_set in tqdm.tqdm(cal_set_collection, desc=desc):

        INP = set()

        # Happens with prob b^2
        # Add 0 and 1 to INP
        if all(cal_set):
            INP = {0, 1}
        # Happens with prob 1 - b^2
        # Add 0 to INP
        else:
            INP = {0}
        INP_size = len(INP)

        # INP has all the values
        if INP_size == 2:
            successful_confidence_counter += 1
            number_of_double_label_predictions += 1
            continue

        # INP has only 0
        test_set_collection = np.random.binomial(L, b)
        # count number of 0s
        number_of_zeros = L - test_set_collection
        successful_prediction_counter = number_of_zeros

        coverage = successful_prediction_counter/L

        if coverage >= 1 - E[i]:
            successful_confidence_counter += 1

    P_S_E = successful_confidence_counter/M
    rate_of_double_label_predictions = number_of_double_label_predictions/M
    rate_of_single_label_predictions = P_S_E - rate_of_double_label_predictions

    P_S_E_array[i] = P_S_E
    rate_of_double_label_predictions_array[i] = rate_of_double_label_predictions
    rate_of_single_label_predictions_array[i] = rate_of_single_label_predictions

# Plot the PAC guarantees
plt.figure(figsize=(8, 6))
# Fill areas under the curve with specified colors
plt.fill_between(E, 0, rate_of_single_label_predictions_array, color='red', alpha=0.3, label='$\mathbb{P}^2(S_E \cap \Gamma^{\epsilon}=\overline{Q})$')
plt.fill_between(E, rate_of_single_label_predictions_array, P_S_E_array, color='blue', alpha=0.3, label='$\mathbb{P}^2(S_E \cap \Gamma^{\epsilon}=\mathbf{Z})$')
# Plot E vs E^2
plt.plot(E, E**2, label="$E^2$", color='black', linewidth=2)
# Plot P_S_E_array
plt.plot(E, P_S_E_array, label="$\mathbb{P}^2(S_E)$", color='red', linewidth=2)
plt.xlabel("$E$")
plt.ylabel("Probability")
plt.title("PAC guarantees using conformal prediction")
# Legend position center left
plt.legend(loc='center left')
plt.grid()
plt.show()