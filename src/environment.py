class Environment:

    def __init__(self):
    def reset(self):  # resets to the start of the day
        # here we need to reselect our value `a` which scales the day's load
    def step(self, num_vms):  # moves us forward one hour
    def get_time(self):  # gets current timesteps in hours from the start of the day
    def get_data_processed(self):
        # I imagine this will be proportional to the number of VMs. Will will check ratio of VM
        # capacity to total data to process later
    def get_data_to_process(self):
        # we want the amount of data during the day to generally look like a*(sin(x) + 2) + n_t,
        # where n_t is some noise centred on 0 and a is a random value that is around 1 to
        # control the "uncertainty"
    def get_availability(self):  # TODO: do spot VMs tell us this?
    def get_current_vms(self):
    def get_on_peak_mins(self):  # on peak VMs * hours
    def get_off_peak_mins(self):  # off peak VMs * hours
    def get_state_vector(self):
        # state is:
        #  - data to process
        #  - data processed
        #  - (data to process - data processed) = data remaining
        #  - time of day (hour / 24)
        #  - VM availability so far (cumulative)
        #  - VM availability right now
        #  - current number of VMs being used
    def get_reward(self, alpha, beta, gamma):
        # reward
        #  - #VM * minutes on peak
        #  - #VM * minutes off peak
        #  - 1 if data is processed by the end of the day else 0
        # weighted by alpha, beta, and gamma respectively