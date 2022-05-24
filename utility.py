'''
Utility functions for marketplace simulations
by Alejandro Cuevas 2021
'''

from bisect import bisect_left

ENC_HISTORY_DIR = '/home/acuevasv/pycharm_projects/marketplace/simulation/rscripts/encounter_histories/'
'''
Create an .inp file for the RMark package using the dictionary format in the simulation
IN: a dictionary of encounter histories of the form {item_id: {day: encounter}} and a title string for the file
OUT: the full filepath
Notes: This function uses the .inp format with no headers and collapses the frequencies. Otherwise, RMark complains
if there are more than 50,000 rows.
'''
def create_enc_history_file(items_enc_histories, filepath):
    enc_history_shortform = {}
    for item, enc in items_enc_histories.items():
        # we need to sort the encounter history
        enc_history = {k: v for k, v in sorted(enc.items())}
        enc_history_short = ''
        for day, value in enc_history.items():
            enc_history_short += str(value)
        if enc_history_short not in enc_history_shortform:
            enc_history_shortform[enc_history_short] = 1
        else:
            enc_history_shortform[enc_history_short] += 1

    with open(filepath, 'w+') as fw:
        for enc_history, freq in enc_history_shortform.items():
            fw.write(str(enc_history+'\t'+str(freq)+'\n'))

    # We create an alternative .inp file which is less sparse by removing encounter histories with <X% frequency
    # and then distribute the encounters uniformly across the other frequencies.
    # This alternative encounter history is helpful to navigate convergence issues due to objects with too little observations
    max_freq = max(enc_history_shortform.values())
    freq_to_distrib = 0
    indexes_to_remove = []
    for enc_history, freq in enc_history_shortform.items():
        if freq < int(max_freq/3):
            freq_to_distrib += freq
            indexes_to_remove.append(enc_history)

    # we delete from the dict
    for i in indexes_to_remove:
        del enc_history_shortform[i]

    # we now redistribute the quantities equally across the other encounter histories
    # this approach is by no means perfect but among the few ways to mitigate the occassional numerical convergence issues
    for enc in enc_history_shortform:
        enc_history_shortform[enc] += int(freq_to_distrib / len(enc_history_shortform))

    filepath_alt = filepath+'.alt'
    with open(filepath_alt, 'w+') as fw:
        for enc_history, freq in enc_history_shortform.items():
            fw.write(str(enc_history+'\t'+str(freq)+'\n'))

    return filepath, filepath_alt

def cprint(print_object, mute_all=False):
    if not mute_all:
        print(print_object)

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

if __name__ == '__main__':
    pass
