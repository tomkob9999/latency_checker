# Latency Checker
# Author: Tomio Kobayashi
# Version 2.0.8
# Updated: 2024/03/10

import requests
            
class latency_checker:
        
    # Function to perform a request and measure latency
    def make_request(url, data, func=None):
        start_time = time.time()
        if func is None:
            response = requests.post(url, data=data)  # Assuming POST request; adjust as needed
        else:
            response = func(url, data)  # Assuming POST request; adjust as needed
        latency = time.time() - start_time
        output_volume = 0
        if response is not None:
            if func is None:
                output_volume = len(response.content)
            else:
                output_volume = len(response)
        return latency, len(data), output_volume

    # Function to simulate different input volumes (customize as per your needs)
    def generate_input_data(size):
        return 'x' * size  # Example: generates a string 'x' multiplied by size

    # Function to perform multiple requests concurrently and measure average latency
    def measure_latency_concurrent(url, conrequest_count, input_size, unit_M=False, func=None, silent=True):
        unit_size = 1024 if not unit_M else 1024 * 1024
        with ThreadPoolExecutor(max_workers=conrequest_count) as executor:
            futures = [executor.submit(latency_checker.make_request, url, latency_checker.generate_input_data(input_size*unit_size), func=func) for _ in range(conrequest_count)]

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
            return average_latency
    def take_averages(url, num_reqs, inp_sizes, unit_M=False, func=None, silent=True):

        # Example usage
        res = []

        for num_req in num_reqs:
            for inp_size in inp_sizes:
                res.append([num_req, inp_size, latency_checker.measure_latency_concurrent(url, num_req, inp_size, unit_M=unit_M, func=func, silent=silent)])
        return res
    
    def measure(url, num_conreqs, inp_sizes, const_thresh=0.1, unit_M=False, use_lasso=False, skip_inverse=True, func=None, silent=True):
        print("")
        print("Measuring relations to latency in accessing", url, "...")
        stats = latency_checker.take_averages(url, num_conreqs, inp_sizes, unit_M=unit_M, func=func, silent=silent)

        mid_num_conreq=num_conreqs[int(len(num_conreqs)/2)] if len(num_conreqs) > 2 else num_conreqs[0]
        mid_inp_size=inp_sizes[int(len(inp_sizes)/2)]  if len(inp_sizes) > 2 else inp_sizes[0]

        if use_lasso:
            relation_finder.find_relations(stats, "", "Latency in secs", cols=["NUMBER OF CONCURRENT REQUESTS", "INPUT SIZE"], const_thresh=const_thresh, skip_inverse=skip_inverse, use_lasso=use_lasso)
        else:
            if len(num_conreqs) > 1:
                    dd = [[d[0], d[2]] for d in stats if d[1] == mid_inp_size]
                    relation_finder.find_relations(dd, "NUMBER OF CONCURRENT REQUESTS", "Latency in secs", const_thresh=const_thresh, skip_inverse=skip_inverse, use_lasso=use_lasso)
            else:
                print("Not enough samples for NUMBER OF CONCURRENT REQUESTS")
            if len(inp_sizes) > 1:
                dd = [[d[1], d[2]] for d in stats if d[0] == mid_num_conreq]
                relation_finder.find_relations(dd, "INPUT SIZE", "Latency in secs", const_thresh=const_thresh, use_lasso=use_lasso)
            else:
                print("Not enough samples for INPUT SIZE")
    
        return stats