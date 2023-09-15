import os
import matplotlib.pyplot as plt

def visualise_data_processed(data_processed):
    hours = [i for i in range(24)]
    plt.plot(hours, data_processed)

    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Data processed')
    plt.title('Data processed over a day')

    # If graphs folder doesn't exist make it
    if not os.path.exists('src/graphs'):
        os.makedirs('src/graphs')
    
    # Save graph to folder
    plt.savefig('src/graphs/data_processed_day.png')