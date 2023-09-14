class Environment:  # TODO

    def __init__(self):
    def reset(self):  # resets to the start of the day
        # here we need to reselect our value `a` which scales the day's load
    def step(self, num_vms):  # moves us forward one hour
    def get_time(self):  # gets current timesteps in hours from the start of the day
    def get_data_to_process(self):
        # we want the amount of data during the day to generally look like a*(sin(x) + 2) + n_t,
        # where n_t is some noise centred on 0 and a is a random value to control the "uncertainty"
        # RETURNS NUM BYTES
    def get_data_processed(self):
        # I imagine this will be proportional to the number of VMs. Will will check ratio of VM
        # capacity to total data to process later
        # RETURNS NUM BYTES
    def get_availability(self):  # do spot VMs even tell us this?
    def get_cum_availability(self):  # cumulative availability
    def get_current_vms(self):
    def get_on_peak_mins(self):  # on peak VMs * hours
    def get_off_peak_mins(self):  # off peak VMs * hours
    def get_state_vector(self):
        #  - data to process
        #  - data processed
        #  - (data to process - data processed) = data remaining
        #  - time of day (hour / 24)
        #  - VM availability right now
        #  - VM availability so far (cumulative)
        #  - current number of VMs being used

        return np.array([
            self.get_data_to_process() / 1_000_000_000,  # we deal in GB
            self.get_data_processed() / 1_000_000_000,
            (self.get_data_to_process() - self.get_data_processed()) / 1_000_000_000,
            self.get_time() / 24,
            self.get_availability(),
            self.get_cum_availability(),
            self.get_current_vms(),
        ])
    def get_reward(self, alpha, beta, gamma):
        #  - #VM * minutes on peak
        #  - #VM * minutes off peak
        #  - 1 if data is processed by the end of the day else 0
        # weighted by alpha, beta, and gamma respectively

        return alpha * self.get_on_peak_mins() \
             + beta * elf.get_off_peak_mins() \
             + gamma * int(self.get_data_to_process() - 128_000_000_000 > self.get_data_processed() \
                           and self.get_time() == 24)  # here we have 128MB slack on data processing requirements