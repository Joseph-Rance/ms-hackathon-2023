import matplotlib.pyplot as plt

def visualise_data_processed(data_processed):
    hours = [i for i in range(24)]
    plt.plot(hours, data_processed)

    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Data processed')
    plt.title('Data processed over a day')

    # Show the plot
    plt.show()