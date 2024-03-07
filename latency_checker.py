# Latency Checker
# Author: Tomio Kobayashi
# Version 2.0.0
# Updated: 2024/03/07
    
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Lasso

import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class latency_checker:
    def find_relations(data, colX, colY, cols=[], const_thresh=0.1, use_lasso=False):
            
        dic_relation = {
            0: ("P", "Proportional Linearly (Y=a*X)"),
            1: ("IP", "Inversely Proportional Linearly (Y=a*(1/X))"),
            2: ("QP", "Proportional Quadruply (Y=a*(X^2))"),
            3: ("IQP", "Inversely Proportional Quadruply (Y=a*(1/X^2))"),
            4: ("SP", "Proportional by Square Root (Y=a*sqrt(X)"),
            5: ("ISP", "Inversely Proportional by Square Root (Y=a*(1/sqrt(X))"),
        }
            
        xcol_size = len(data[0])-1
#         xcol_size = 1
        for i in range(xcol_size):
            for row in data:
                row.insert(-1, 1/row[i])
        for i in range(xcol_size):
            for row in data:
                row.insert(-1, row[i] ** 2)
        for i in range(xcol_size):
            for row in data:
                row.insert(-1, 1/row[i] ** 2)
        for i in range(xcol_size):
            for row in data:
                row.insert(-1, np.sqrt(row[i]))
        for i in range(xcol_size):
            for row in data:
                row.insert(-1, 1/np.sqrt(row[i]))
        
        
        if use_lasso:
            
            model = Lasso()
            X_train = [r[:-1]for r in data]
#             print("X_train", X_train)
            Y_train = [r[-1]for r in data]
#             print("Y_train", Y_train)
            model.fit(X_train, Y_train)

            print(f"Relation between {colY} and {colX}:")
#             print("  Intersect:", model.intercept_)
#             print("  Coeffeicients:", model.coef_)
            print("  Coeffeicients:")
            for i, c in enumerate(model.coef_):
                if np.abs(c) > 0.0000001:
                    print("    ", cols[int(i/6)] if len(cols) > 0 else "    Col" + str(int(i/6)), ":", dic_relation[i%6][1], round(c, 10))
            predictions = model.predict(X_train)
            r2 = r2_score(Y_train, predictions)
            print("  R2:", round(r2, 5))
                
            return [r2, model.coef_, model.intercept_], None
        
        else:
            # print(data)
            stats = []
            for i in range(len(data[0])-1):
                X_train = [[row[i]] for row in data]
                Y_train = [row[-1] for row in data]
                # Creating a linear regression model
                model = LinearRegression()
                model.fit(X_train, Y_train)
                predictions = model.predict(X_train)
                r2 = r2_score(Y_train, predictions)

                # Displaying the model parameters
                stats.append([r2, model.coef_[0], model.intercept_, i])

                most_fit = sorted(stats, key=lambda x: np.abs(x[0]), reverse=True)[0]
    #         print("const_thresh", const_thresh)
            if most_fit[1] < const_thresh:
                print(f"{colY} is CONSTANT to {colX} with constant value of {most_fit[2]:.5f} with assurance (R2) of {most_fit[0]*100:.2f}%")
            else:
                print(f"{colY} is {dic_relation[most_fit[3]][1].upper()} to {colX} with coefficent of {most_fit[1]:.6} with intercept of {most_fit[2]:.6f} with assurance (R2) of {most_fit[0]*100:.2f}%")

            return most_fit, dic_relation[most_fit[3]]
    

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
    def measure(url, num_conreqs, inp_sizes, const_thresh=0.1, unit_M=False, use_lasso=False, silent=True):
        print("")
        print("Measuring relations to latency in accessing", url, "...")
        stats = latency_checker.take_averages(url, num_conreqs, inp_sizes, unit_M=unit_M, silent=silent)

#         print("take_averages FINISHED")
        mid_num_conreq=num_conreqs[int(len(num_conreqs)/2)] if len(num_conreqs) > 2 else num_conreqs[0]
        mid_inp_size=inp_sizes[int(len(inp_sizes)/2)]  if len(inp_sizes) > 2 else inp_sizes[0]

        if use_lasso:
            latency_checker.find_relations(stats, "", "Latency in secs", cols=["NUMBER OF CONCURRENT REQUESTS", "INPUT SIZE"], const_thresh=const_thresh, use_lasso=use_lasso)
        else:
            if len(num_conreqs) > 1:
                    dd = [[d[0], d[2]] for d in stats if d[1] == mid_inp_size]
            #         latency_checker.find_relations(dd, "Number of Requests", "Latency in secs", const_thresh=const_thresh)
                    latency_checker.find_relations(dd, "NUMBER OF CONCURRENT REQUESTS", "Latency in secs", const_thresh=const_thresh, use_lasso=use_lasso)
            else:
                print("Not enough samples for NUMBER OF CONCURRENT REQUESTS")
            if len(inp_sizes) > 1:
                dd = [[d[1], d[2]] for d in stats if d[0] == mid_num_conreq]
        #         latency_checker.find_relations(dd, "Input Size", "Latency in secs", const_thresh=const_thresh)
                latency_checker.find_relations(dd, "INPUT SIZE", "Latency in secs", const_thresh=const_thresh, use_lasso=use_lasso)
            else:
                print("Not enough samples for INPUT SIZE")
    
        return stats

    
url = "http://example.com/api"  # Endpoint you are testing
num_conreqs = [1, 5, 10, 20]
# num_conreqs = [1]
# maxsize = 10000
# inp_sizes = [i for i in range(1, maxsize, int(maxsize/20))]
# stats = latency_checker.measure(url, num_conreqs, inp_sizes, const_thresh=0.0001, unit_M=False)
maxsize = 50
inp_sizes = [i for i in range(1, maxsize, int(maxsize/5))]
stats = latency_checker.measure(url, num_conreqs, inp_sizes, const_thresh=0.1, unit_M=True, silent=True, use_lasso=True)