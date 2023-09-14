import numpy as np
import math

class Environment:

    def __init__(self, peak_hours=list(range(8, 22))):
        self.reset()
        self.peak_hours = peak_hours

    def reset(self):  # resets to the start of the day
        self.time = self.data_to_process = self.data_processed = self.cum_availability = \
            self.on_peak_vm_hours = self.off_peak_vm_hours = 0
        self.current_vms = -1
        self.availability = int(np.random.normal(loc=5, scale=5))
        self.a = np.random.normal(loc=1, scale=0.05)

    def step(self, num_vms):  # moves us forward one hour
        
        self.availability = max(0, int(np.random.normal(loc=self.availability, scale=5)))
        self.cum_availability += self.availability

        self.current_vms = min(self.availability, num_vms)
        if self.time in self.peak_hours:
            self.on_peak_vm_hours += self.current_vms
        else:
            self.off_peak_vm_hours += self.current_vms

        scale = 4_000_000_000 / 3 # Based on the fact of 400GB of data being processed (previous data)
        noise = np.random.normal(loc=0, scale=0.1)
        self.data_to_process = scale * self.a * (math.sin(self.time * (math.pi / 12) - 2) + 2) + noise
        self.data_processed = min(self.data_processed + self.current_vms * 150_000_000_000, self.data_to_process)

        self.time += 1  # time corresponds to the hour that the step will start on

    def get_time(self):  # gets current timesteps in hours from the start of the day
        return self.time

    def get_data_to_process(self):  # in bytes
        return self.data_to_process

    def get_data_processed(self):
        return self.data_processed

    def get_availability(self):  # availability of VMs in last hour
        return self.availability

    def get_cum_availability(self):  # cumulative availability
        return self.cum_availability

    def get_current_vms(self):
        return self.current_vms

    def get_on_peak_vm_hours(self):  # on peak VMs * hours
        return self.on_peak_vm_hours

    def get_off_peak_vm_hours(self):  # off peak VMs * hours
        return self.off_peak_vm_hours

    def get_state_vector(self):
        #  - data to process
        #  - data processed
        #  - time of day (hour / 24)
        #  - VM availability right now
        #  - VM availability so far (cumulative)
        #  - current number of VMs being used

        return [
            self.get_data_to_process() / 1_000_000_000,  # we deal in GB
            self.get_data_processed() / 1_000_000_000,
            self.get_time() / 24,
            self.get_availability(),
            self.get_cum_availability(),
            self.get_current_vms(),
        ]
    def get_reward(self, alpha, beta, gamma):
        #  - #VM * minutes on peak
        #  - #VM * minutes off peak
        #  - 1 if data is processed by the end of the day else 0
        # weighted by alpha, beta, and gamma respectively

        return alpha * self.get_on_peak_vm_hours() \
             + beta * self.get_off_peak_vm_hours() \
             + gamma * int(self.get_data_to_process() - 128_000_000_000 > self.get_data_processed() \
                           and self.get_time() == 24)  # here we have 128MB slack on data processing requirements