class Environment:

    def __init__(self):
    def reset(self):  # resets to the start of the day
    def step(self, num_vms):  # moves us forward one hour
    def get_time(self):  # gets current timesteps in hours from the start of the day
    def get_data_processed(self):
    def get_data_to_process(self):
        # we want the amount of data during the day to generally look like a*(sin(x) + 2) + n_t,
        # where n_t is some noise centred on 0 and a is a random value that is around 1 to
        # control the "uncertainty"
    def get_availability(self):
    def get_current_vms(self):
    def get_on_peak_mins(self):  # on peak VMs * hours
    def get_off_peak_mins(self):  # off peak VMs * hours