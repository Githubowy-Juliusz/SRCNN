def pretty_elapsed_time(time_elapsed: float):
	hours = int(time_elapsed / 3600)
	minutes = int(time_elapsed / 60) % 60
	seconds = int(time_elapsed) % 60
	ms = int((time_elapsed % 1) * 1000)
	return f"{hours:02}:{minutes:02}:{seconds:02}.{ms:04}"