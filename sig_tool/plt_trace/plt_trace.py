import matplotlib.pyplot as plt

def empty():
	pass


try PLOT_TRACE:
	plot_trace = plt.plot
	figure_trace = plt.figure	
except:
	plot_trace = empty
	figure_trace = empty

