import time


def print_time(start):
    """
    Print the elapsed time from a specific starting point.
    :param start: Starting time.
    """
    if (time.time()-start)/60 > 1:
        elapsed_time = divmod(time.time()-start, 60)
        if elapsed_time[0] > 1:
            print("\nElapsed time: %.0f minutes, %.2f seconds" % (elapsed_time[0], elapsed_time[1]))
        else:
            print("\nElapsed time: %.0f minute, %.2f seconds" % (elapsed_time[0], elapsed_time[1]))
    else:
        print("\nElapsed time: %.2f seconds" % (time.time()-start))
    print("#"*19 + "\n" + time.strftime("%d/%m/%Y %H:%M:%S") + "\n" + "#"*19 + "\n")
