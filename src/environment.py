import numpy as np

class Environment:  # TODO: lots of made up values that need to be properly set!!!

    def __init__(self, peak_hours=range(8, 22)):
        self.reset()
        self.peak_hours = peak_hours

    def reset(self):  # resets to the start of the day
        self.time = self.data_to_process = self.data_processed = self.cum_availability = \
            self.on_peak_vm_hours = self.off_peak_vm_hours = 0
        self.current_vms = -1
        self.availability = int(np.random.normal(loc=5, scale=5))
        self.a = np.random.normal(loc=1, scale=0.05)

    def step(self, num_vms):  # moves us forward one hour (TODO)
        # we want the amount of data during the day to generally look like scale*self.a*(sin(x) + 2) + n_t,
        # where n_t is some noise centred on 0 and a is a random value to control the "uncertainty"
        # hour 0 is midnight so make sure load lines up!

        # I imagine data_processed will be proportional to the number of VMs. Will will check ratio of VM
        # capacity to total data to process later

        # all data measurements are in bytes

        self.availability = int(np.random.normal(loc=5, scale=5))

    def get_time(self):  # gets current timesteps in hours from the start of the day
        return self.time

    def get_data_to_process(self):
        return self.data_to_process

    def get_data_processed(self):
        return self.data_processed

    def get_availability(self):  # do spot VMs even tell us this?
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