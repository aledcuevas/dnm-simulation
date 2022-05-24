'''
Statistics functions for marketplace simulation
created by Alejandro Cuevas 2021
'''
import subprocess
import numpy as np
import math

RSCRIPT_DIR = '/home/acuevasv/pycharm_projects/marketplace/simulation/rscripts/'

'''
Jolly Seber estimate using RMark package
We call the R script from python and read the output file to get the result
The markscript.R accepts two params: the .inp filepath and the output file
We read from the output file and return that value as our JS estimate.
'''
def jolly_seber_estimate(encounter_history_file, mark_file_ct, r_simulation_env):
    RMARK_SCRIPT = RSCRIPT_DIR + 'markscript.R'

    ESTIMATE_FILE = r_simulation_env + 'estimate.out'
    RMARK_LOG = r_simulation_env + 'markscript-log.out'
    R_DATA = r_simulation_env + 'Rdata_previous_analysis.env'
    MARK_OUTPUT = r_simulation_env + 'mark' + str(mark_file_ct)

    if mark_file_ct == 0:
        with open(ESTIMATE_FILE, 'w+') as fw:
            fw.write('')
        # If we don't have a previous R_DATA file, we just pass NULL and the script will run the model for the first time
        prev_model = 'NULL'
    else:
        # We check if the previous run was problematic. If it was, we don't want to seed it into the new run.
        with open(ESTIMATE_FILE, 'r') as fr:
            js_estimate = fr.readline()
            js_estimate = round(float(js_estimate.strip('\n')))
            if js_estimate < 0:
                prev_model = 'NULL'
            else:
                prev_model = 'Continue'

    subprocess.check_call([RMARK_SCRIPT, encounter_history_file, ESTIMATE_FILE, prev_model, R_DATA, MARK_OUTPUT],
                          stdout=open(RMARK_LOG,'wb'), stderr=open(RMARK_LOG,'wb'))

    with open(ESTIMATE_FILE, 'r') as fr:
        js_estimate = fr.readline()

    if round(float(js_estimate.strip('\n'))) > 1000000:
        js_estimate = -1
    else:
        js_estimate = round(float(js_estimate.strip('\n')))

    return js_estimate

'''
Lincoln-Petersen Abundance Estimation
Grabs the encounter history and uses the last two dates to compute the estimate
- INPUT: simulation encounter history
- OUTPUT: LP abundance estimation
'''
def lp_estimate(encounter_history):
    # N = Kn/k
    # n (animals captured on first visit)
    first_encounters = 0
    # K (animals captured on second visit)
    second_encounters = 0
    # k (recaptures)
    recaptures = 0
    for item_id, enc_history in encounter_history.items():
        enc = {k: v for k, v in sorted(enc_history.items())}
        first_mark = list(enc.values())[-2]
        second_mark = list(enc.values())[-1]
        first_encounters += first_mark
        second_encounters += second_mark
        if first_mark == second_mark == 1:
            recaptures += 1

    estimate = (first_encounters * second_encounters) / recaptures

    return estimate

'''
Schnabel Abundance Estimation
Grabs the encounter history file and computes the Schnabel estimator
- INPUT: .inp file (for simplicity
- OUTPUT: Schnabel abundance estimation
NOTE: Requires a minimum of 3 samples
Formula: N = \sum^m_i=1 M_i C_i / \sum_^m_i=1 R_i
Reference: http://derekogle.com/fishR/examples/oldFishRVignettes/MRClosed.pdf
'''
def schnabel_estimate(encounter_history_file):
    enc_history_list = []
    freq_list = []
    # read our .inp file of the form '0101001\t34' and create an array of encounter histories and a vector of frequencies
    with open(encounter_history_file, 'r') as fr:
        for line in fr.readlines():
            enc_history, freq = line.strip('\n').split('\t')
            enc_array = np.asarray([int(i) for i in enc_history])
            enc_history_list.append(enc_array)
            freq_list.append(int(freq))
    enc_history_list = np.asarray(enc_history_list)
    reencounter_array = enc_history_list

    # we create an array of total individuals tagged per occasion
    # C_i
    total_capped = []
    for i, col in enumerate(enc_history_list.T):
        # each column corresponds to a 'site visit' and their row position corresponds to a frequency so we can do the dot product
        total_capped.append(np.dot(col, freq_list))

    # we now create an array of recaptures. Note: python converts enc_history_list to recaptures as well by reference
    for i, row in enumerate(enc_history_list):
        new_row = True
        for j, cell in enumerate(row):
            if new_row is True and cell == 1:
                reencounter_array[i][j] = 0
                new_row = False
            else:
                pass

    # we create a list of recaptured items per occasion
    # R_i
    total_recapped = []
    # the difference gives us the total number of new individuals per occasion
    total_new = []
    for i, col in enumerate(reencounter_array.T):
        # this is the same process as above but now done on the matrix of recaptures
        recap_total = np.dot(col, freq_list)
        total_recapped.append(recap_total)
        # the vector of new captures is not used in calculation but good for debugging
        total_new.append(total_capped[i] - recap_total)

    # M_i
    total_tagged = []
    # total animals tagged in each occassion
    # from 1st -> 2nd occassion is just equal to the number of animals caught
    for i, new in enumerate(total_new):
        if i == 0:
            total_tagged.append(0)
        else:
            total_tagged.append(total_new[i-1] + total_tagged[i-1])

    #return total_capped, total_recapped, total_new, total_tagged
    # estimate is given with the Chapman modification
    estimate = np.dot(total_capped, total_tagged)/ (np.sum(total_recapped)+1)
    return estimate

'''
Test reference: http://derekogle.com/fishR/examples/oldFishRVignettes/MRClosed.pdf &
                https://derekogle.com/FSA/index.html
Dataset = PikeNYPartial1 from library(FSAdata)
'''
def test_schnabel_estimator():
    '''
    /home/acuevasv/pycharm_projects/marketplace/simulation/rscripts/tests/pikedata.inc
    Data Summary :
    - 1100 3
    - 1010 2
    - 1001 1
    - 1000 21
    - 0110 2
    - 0101 1
    - 0100 12
    - 0011 2
    - 0010 8
    - 0001 5
    '''
    estimate = schnabel_estimator('./rscripts/tests/pikedata.inp')
    '''
        n   m   R   M 
    1   27  0   27  0 
    2   18  3   18  27
    3   14  4   14  42
    4   9   4   0   52
    
    Where:
    - n is total_capped per occasion (also labeled C_i)
    - m is total_recapped per occasion (also labeled R_i)
    - M is total_tagged per occasion (labeled M_i)
    - R is the number of tagged individuals re-released into the population (in our case, we always release them back)
    '''
    # Note that the reference above uses something called 'the Chapman modification to the Schnabel method' which means they add
    # +1 to the denominator
    assert math.floor(estimate == 82)

if __name__ == '__main__':
    pass


'''
    # we will iterate through our encounter list and 0 out the first 1 in each row, the leftovers are all recaptures
    reencounter_array = enc_history_list
    # number of recaptures
    denominator = 0
    for i, row in enumerate(enc_history_list):
        new_row = True
        for j, cell in enumerate(row):
            if new_row is True and cell == 1:
                reencounter_array[i][j] = 0
                new_row = False
            else:
                pass
'''
