# Latency Checker
# Author: Tomio Kobayashi
# Version 2.0.3
# Updated: 2024/03/08
    
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Lasso

import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats as statss
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class latency_checker:
    
    def find_relations(data, colX, colY, cols=[], const_thresh=0.1, skip_inverse=True, use_lasso=False):
        
        z_scores = np.abs(statss.zscore(np.array([np.array(row) for row in data])))
#         print("z_scores", z_scores)
        threshold = 3
        outlier_indices = np.where(z_scores > threshold)[0]
    
#         print("outlier_indices", outlier_indices)
#         plt.figure(figsize=(10, 6))
#         sns.scatterplot(data=pdata, x=iris.feature_names[0], y=iris.feature_names[1], hue='species', style='species', palette='bright')
#         print("data", data)

        if len(outlier_indices) > 0:
            print("*")
            print(f"OUTLIER FOUND at: {outlier_indices}")
            for i in outlier_indices:
                print(data[i])
            print("*")

        if use_lasso:
            dic_relation = {
                0: ("P", "Proportional Linearly (Y=a*X)"),
                1: ("IP", "Inversely Proportional Linearly (Y=a*(1/X))"),
                2: ("QP", "Proportional Quadruply (Y=a*(X^2))"),
                3: ("IQP", "Inversely Proportional Quadruply (Y=a*(1/X^2))"),
                4: ("SP", "Proportional by Square Root (Y=a*sqrt(X)"),
                5: ("ISP", "Inversely Proportional by Square Root (Y=a*(1/sqrt(X))"),
            }
            num_incs = 6

            if skip_inverse:
                dic_relation = {
                    0: ("P", "Proportional Linearly (Y=a*X)"),
                    1: ("QP", "Proportional Quadruply (Y=a*(X^2))"),
                    2: ("SP", "Proportional by Square Root (Y=a*sqrt(X)"),
                }
                num_incs = 3

            xcol_size = len(data[0])-1
            if not skip_inverse:
                for i in range(xcol_size):
                    for row in data:
                        row.insert(-1, 1/row[i])
            for i in range(xcol_size):
                for row in data:
                    row.insert(-1, row[i] ** 2)
            if not skip_inverse:
                for i in range(xcol_size):
                    for row in data:
                        row.insert(-1, 1/row[i] ** 2)
            for i in range(xcol_size):
                for row in data:
                    row.insert(-1, np.sqrt(row[i]))
            if not skip_inverse:
                for i in range(xcol_size):
                    for row in data:
                        row.insert(-1, 1/np.sqrt(row[i]))
            
            model = Lasso()
            X_train = [r[:-1]for r in data]
            Y_train = [r[-1]for r in data]
            model.fit(X_train, Y_train)

            print(f"Relation to {colY}")
            print("  Intersect:", model.intercept_)
#             print("  Coeffeicients:", model.coef_)
            print("  Coeffeicients:")
    
            for i, c in enumerate(model.coef_):
                if np.abs(c) > 0.0000001:
                    print("    ", cols[int(i/num_incs)] if len(cols) > 0 else "    Col" + str(int(i/num_incs)), ":", dic_relation[i%num_incs][1], round(c, 10))
            predictions = model.predict(X_train)
#             print("predictions", predictions)
            r2 = r2_score(Y_train, predictions)
            print("  R2:", round(r2, 5))
        
#             print("cols", cols)
            for i in range(len(cols)):
#                 print("i", i)
                pdata = [[row[i], row[-1]] for row in data]
                df = pd.DataFrame(pdata, columns=[cols[i], colY])
                plt.title("Scatter Plot of " + cols[i] + " and " + colY)
                plt.scatter(data=df, x=cols[i], y=colY)
                plt.figure(figsize=(3, 2))
                plt.show()
                
            return [r2, model.coef_, model.intercept_], None
        
        else:
        # Fit a polynomial of the specified degree to the data
            X_train = [r[0]for r in data]
            Y_train = [r[-1]for r in data]
            degree = 2
            coefficients = np.polyfit(X_train, Y_train, degree)
            # Create a polynomial from the coefficients
            polynomial = np.poly1d(coefficients)
            predictions = np.polyval(coefficients, X_train)
            
            r2 = r2_score(Y_train, predictions)
            
            if polynomial.coeffs[0] < const_thresh and polynomial.coeffs[1] < const_thresh :
                print(f"{colY} is CONSTANT to {colX} with constant value of {polynomial.coeffs[2]:.5f} with confidence level (R2) of {r2*100:.2f}%")
            else:
                # Generate the polynomial equation as a string
                equation_terms = []
                
                # Equation parameters
                mm = 0
                m = 0  # slope
                b = 0  # y-intercept
                for i, coeff in enumerate(polynomial.coeffs):
                    degree = polynomial.order - i
                    if degree > 1:
                        equation_terms.append(f"{coeff:.3f}x^{degree}")
                        mm = coeff
                    elif degree == 1:
                        equation_terms.append(f"{coeff:.3f}x")
                        m = coeff
                    else:
                        equation_terms.append(f"{coeff:.3f}")
                        b = coeff
                        
                polynomial_equation = " + ".join(equation_terms).replace("+ -", "- ")
#                 print("polynomial_equation", polynomial_equation)
                print(f"{colY} is related to {colX} by equation of [{polynomial_equation}] with confidence level (R2) of {r2*100:.2f}%")
    
                pdata = [[row[0], row[-1]] for row in data]
                df = pd.DataFrame(pdata, columns=[colX, colY])
                plt.title("Scatter Plot of " + colX + " and " + colY)
    #             sns.scatterplot(data=df, x=colX, y=colY)
                plt.scatter(data=df, x=colX, y=colY)
                # Generate x values for the line
                x_line = np.linspace(min(X_train), max(X_train), 10000)  # 100 points from min to max of scatter data
                # Calculate y values based on the equation of the line
                y_line = [mm*x**2 + x * m + b for x in x_line]
                # Plot the line
                plt.plot(x_line, y_line, color='red', label='Line: ' + polynomial_equation)
                plt.figure(figsize=(3, 2))
                plt.show()

                return [r2, polynomial.coeffs], None
    
        
    # Function to perform a request and measure latency
    def make_request(url, data):
        start_time = time.time()
        response = requests.post(url, data=data)  # Assuming POST request; adjust as needed
        latency = time.time() - start_time
        output_volume = len(response.content)
        return latency, len(data), output_volume

    # Function to simulate different input volumes (customize as per your needs)
    def generate_input_data(size):
        return 'x' * size  # Example: generates a string 'x' multiplied by size

    # Function to perform multiple requests concurrently and measure average latency
    def measure_latency_concurrent(url, conrequest_count, input_size, unit_M=False, silent=True):
#         print("measure_latency_concurrent IN")
        unit_size = 1024 if not unit_M else 1024 * 1024
#         print("unit_size", unit_size)
#         print("input_size", input_size)
        with ThreadPoolExecutor(max_workers=conrequest_count) as executor:
            futures = [executor.submit(latency_checker.make_request, url, latency_checker.generate_input_data(input_size*unit_size)) for _ in range(conrequest_count)]

            total_latency = 0
            total_input_volume = 0
            total_output_volume = 0
            for future in as_completed(futures):
                latency, input_volume, output_volume = future.result()
#                     print("Responded", latency, input_volume)
                total_latency += latency
                total_input_volume += input_volume
                total_output_volume += output_volume

            average_latency = total_latency / conrequest_count
            average_input_volume = total_input_volume / conrequest_count
            average_output_volume = total_output_volume / conrequest_count

            if not silent:
                print(f"Average Latency: {average_latency:.4f} seconds for concurrency {conrequest_count} and input size {average_input_volume:.2f} and output size {average_output_volume:.2f}")
    #         print(f"Average Input Volume: {average_input_volume} bytes")
    #         print(f"Average Output Volume: {average_output_volume} bytes")
            return average_latency
    def take_averages(url, num_reqs, inp_sizes, unit_M=False, silent=True):

        # Example usage
        res = []

        for num_req in num_reqs:
            for inp_size in inp_sizes:
                res.append([num_req, inp_size, latency_checker.measure_latency_concurrent(url, num_req, inp_size, unit_M=unit_M, silent=silent)])
        return res
#     number of concurrent requests
#     input data size in kilobytes to send
    def measure(url, num_conreqs, inp_sizes, const_thresh=0.1, unit_M=False, use_lasso=False, skip_inverse=True, silent=True):
        print("")
        print("Measuring relations to latency in accessing", url, "...")
        stats = latency_checker.take_averages(url, num_conreqs, inp_sizes, unit_M=unit_M, silent=silent)

#         print("take_averages FINISHED")
        mid_num_conreq=num_conreqs[int(len(num_conreqs)/2)] if len(num_conreqs) > 2 else num_conreqs[0]
        mid_inp_size=inp_sizes[int(len(inp_sizes)/2)]  if len(inp_sizes) > 2 else inp_sizes[0]

        if use_lasso:
            latency_checker.find_relations(stats, "", "Latency in secs", cols=["NUMBER OF CONCURRENT REQUESTS", "INPUT SIZE"], const_thresh=const_thresh, skip_inverse=skip_inverse, use_lasso=use_lasso)
        else:
            if len(num_conreqs) > 1:
                    dd = [[d[0], d[2]] for d in stats if d[1] == mid_inp_size]
#                     print("dd", dd)
            #         latency_checker.find_relations(dd, "Number of Requests", "Latency in secs", const_thresh=const_thresh)
                    latency_checker.find_relations(dd, "NUMBER OF CONCURRENT REQUESTS", "Latency in secs", const_thresh=const_thresh, skip_inverse=skip_inverse, use_lasso=use_lasso)
            else:
                print("Not enough samples for NUMBER OF CONCURRENT REQUESTS")
            if len(inp_sizes) > 1:
                dd = [[d[1], d[2]] for d in stats if d[0] == mid_num_conreq]
#                 print("dd", dd)
        #         latency_checker.find_relations(dd, "Input Size", "Latency in secs", const_thresh=const_thresh)
                latency_checker.find_relations(dd, "INPUT SIZE", "Latency in secs", const_thresh=const_thresh, use_lasso=use_lasso)
            else:
                print("Not enough samples for INPUT SIZE")
    
        return stats
