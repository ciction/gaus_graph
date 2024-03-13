from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import random
import math
import numpy as np
from scipy.stats import norm
import time

import random
from collections import Counter

app = Flask(__name__, static_folder='templates/static')  # Specify the static folder location


# Function to generate random numbers with Gaussian distribution
def generate_gaussian_random(start=150, end=210, mu=None, sigma=None):
    # Set the mean (mu) to the middle of the range if not provided
    if mu is None:
        mu = (start + end) / 2
    
    # Set standard deviation (sigma) to 1/6 of the range if not provided
    if sigma is None:
        sigma = (end - start) / 10
    
    while True:
        # Generate a random number with Gaussian distribution
        number = random.gauss(mu, sigma)
        # If the number is within the desired range, return it
        if start <= number <= end:
            return number

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        # Generate a random number
        random_number = generate_gaussian_random()
        # Return the number to the web page
        # generate_Graph()
        return render_template('index.html', random_number=random_number)



def generate_Graph():
    # Generate 1000 random numbers following the Gaussian distribution
    num_samples = 100000

    start = time.time()
    gaussian_array = [generate_gaussian_random() for _ in range(num_samples)]
    
    # Access the element at the random index
    random_index = random.randint(0, num_samples - 1)  # Generate a random index within the array bounds
    weighted_random_value = gaussian_array[random_index]
    
    end = time.time()

    execution_time = end - start
    print(f"Execution time: {execution_time:.6f} seconds")

    # Define 
    mu = 180
    sigma = 30
    # Generate weighted random values using inverse transform sampling
    start = time.time()
    weighted_random_values = norm.ppf(np.random.rand(num_samples), loc=mu, scale=sigma)
    end = time.time()

    
    execution_time = end - start
    print(f"Execution time: {execution_time:.6f} seconds")



    # Create a histogram to visualize the distribution of the data
    plt.hist(gaussian_array, bins=50, edgecolor='black')
    plt.xlabel('Generated Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Gaussian Random Numbers (1000 Samples)')
    plt.grid(True)
    plt.show()
    # Save the plot as a SVG image
    plt.savefig('./templates/static/gaussian_distribution.svg')  # Replace 'gaussian_distribution.svg' with your desired filename
    # Display the plot (optional)
    plt.show()



@app.route('/generate-plot', methods=['GET', 'POST'])
def generate_plot():
    try:
        mu = float(request.form['mu']) # Get user-provided mean from the form
        sigma = float(request.form['sigma']) # Get user-provided standard deviation from the form
    except ValueError:
        # Handle the case where conversion fails (e.g., non-numeric input)
        return "Invalid input. Please enter numbers for mu and sigma."

    # Generate data and create plot using Matplotlib
    data = [generate_gaussian_random(mu= mu, sigma=sigma) for _ in range(1000)]
    plt.hist(data, bins=50, edgecolor='black')
    plt.xlabel('Generated Values')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Gaussian Random Numbers (mu={mu}, sigma={sigma})')
    plt.grid(True)

    # Convert plot to SVG using BytesIO
    from io import BytesIO
    fig_output = BytesIO()
    plt.savefig(fig_output, format='svg')
    plt.close()  # Close the plot to avoid memory issues

    # Return the SVG image data as a string
    return fig_output.getvalue().decode('utf-8')

# # Example usage
# for _ in range(10):
#     print(generate_gaussian_random())




if __name__ == '__main__':
    # app.run(debug=True)
    generate_Graph()
