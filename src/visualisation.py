import os
import matplotlib.pyplot as plt

def visualise_data_processed(data_processed):
    plt.plot(data_processed)

    plt.xlabel('Time')
    plt.ylabel('Data processed')
    plt.title('Data processed over a day')

    if not os.path.exists('graphs'):
        os.makedirs('graphs')
    
    plt.savefig('graphs/data_processed_day.png')
    plt.close()